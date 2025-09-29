
import argparse
import getpass
import sys
import os
import json
from datetime import datetime, timezone



import boto3
from botocore.exceptions import NoCredentialsError, ClientError, NoRegionError, EndpointConnectionError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", type=str, default=None, help="AWS region to inspect (default: from AWS credentials/config)")
    p.add_argument("--output", type=str, default=None, help="Output file path (default: print to stdout)")
    p.add_argument("--format", type=str, choices=["json", "table"], default="json", help="Output format: json or table (default: json)")
    
    return p.parse_args()



def get_region(args):
    # 1. 命令列參數
    if args.region:
        return args.region

    # 2. 環境變數
    env_region = os.getenv("AWS_DEFAULT_REGION")
    if env_region:
        return env_region

    # 3. 沒有設定 → 跳出
    print("Error: No region specified. Please use --region or set AWS_DEFAULT_REGION.")
    sys.exit(1)


def get_session(region):

    
    # 1. 先問使用者
    # AWS CLI credentials
    key = input("Enter AWS Access Key ID (press Enter to skip): ").strip()
    secret = None
    if key:
        secret = getpass.getpass("Enter AWS Secret Access Key: ").strip()

    # 2. 如果使用者有輸入，就嘗試驗證
    if key and secret:
        try:
            session = boto3.Session(
                aws_access_key_id=key,
                aws_secret_access_key=secret,
                region_name=region
            )
            # 驗證身份
            sts = session.client("sts")
            sts.get_caller_identity()
            print("Authentication OK (manual input)")
            
            return session
        
        except Exception as e:
            print(f" Manual credentials failed: {e}. Will try env/CLI next...")
    


    # 3. 環境變數 / CLI fallback
    try:
        session = boto3.Session(region_name=region)
        sts = session.client("sts")
        sts.get_caller_identity()
        print("Authentication OK (environment/CLI)")
        
        return session
    
    except NoCredentialsError:
        print("No AWS credentials found. Set env vars or run 'aws configure'.")
        sys.exit(1)
    except NoRegionError:
        print("No region specified. Use --region or set AWS_DEFAULT_REGION.")
        sys.exit(1)
    except EndpointConnectionError as e:
        print(f"Endpoint/region error: {e}")
        sys.exit(1)
    except ClientError as e:
        print(f"Authentication failed (STS): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected auth error: {e}")
        sys.exit(1)




def _iso(dt):
    if not dt:
        return None
    # boto3 回傳的是 tz-aware datetime（UTC）；統一輸出成 ISO8601 "Z"
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def collect_iam_users(session: boto3.Session):
   
   
    iam = session.client("iam")
    resp = iam.list_users()
    
    iam_users = []

    paginator = iam.get_paginator("list_users")
    for page in paginator.paginate():
        for u in resp.get("Users", []):

            user_name = u.get("UserName")
            user_id = u.get("UserId")
            user_arn = u.get("Arn")
            create_date = _iso(u.get("CreateDate"))

            # 1) 取 PasswordLastUsed 當 last_activity（若無可能為 None）
            last_activity = None
            try:
                gu = iam.get_user(UserName=user_name)
                last_activity = _iso(gu.get("User", {}).get("PasswordLastUsed"))
            
            except ClientError as e:
                # 若無權限或該使用者沒有 console 密碼，這個欄位可能不存在；忽略即可
                last_activity = None

            # 2) 取附掛的 managed policies
            attached_policies = []

            lp = iam.list_attached_user_policies(UserName=user_name)

            pol_paginator = iam.get_paginator("list_attached_user_policies")
            for pol_page in pol_paginator.paginate(UserName=user_name):
                for p in lp.get("AttachedPolicies", []):
                    attached_policies.append({
                        "policy_name": p.get("PolicyName"),
                        "policy_arn": p.get("PolicyArn"),
                    })



            iam_users.append({
                "username": user_name,
                "user_id": user_id,
                "arn": user_arn,
                "create_date": create_date,
                "last_activity": last_activity,  # PasswordLastUsed（可能為 None）
                "attached_policies": attached_policies,
            })


    return iam_users



def collect_ec2_instances(session):

    ec2 = session.client("ec2")

    # 1) 先把所有 instances 列出來（含分頁）
    instances_raw = []
    paginator = ec2.get_paginator("describe_instances")
    for page in paginator.paginate():
        for res in page.get("Reservations", []):
            for inst in res.get("Instances", []):
                instances_raw.append(inst)

    if not instances_raw:
        return []  # 沒有任何 EC2


    # 2) 蒐集所有 AMI ID，等等一次查出名字
    # Amazon Machine Image
    ami_ids = {inst.get("ImageId") for inst in instances_raw if inst.get("ImageId")}
    ami_name_map = {}
    if ami_ids:
        # describe_images 最多一次 100 張，做個簡易分批
        ami_ids = list(ami_ids)
        for i in range(0, len(ami_ids), 100):
            chunk = ami_ids[i:i+100]
            imgs = ec2.describe_images(ImageIds=chunk).get("Images", [])
            for img in imgs:
                ami_name_map[img["ImageId"]] = img.get("Name")

    # 3) 組成乾淨輸出
    ec2_instances = []

    for inst in instances_raw:
        instance_id   = inst.get("InstanceId")
        instance_type = inst.get("InstanceType")
        state         = (inst.get("State") or {}).get("Name")
        public_ip     = inst.get("PublicIpAddress")
        private_ip    = inst.get("PrivateIpAddress")
        az            = (inst.get("Placement") or {}).get("AvailabilityZone")
        launch_time   = _iso(inst.get("LaunchTime"))
        ami_id        = inst.get("ImageId")
        ami_name      = ami_name_map.get(ami_id)

        # SecurityGroups: 取 group-id 清單
        sgs = []
        for sg in inst.get("SecurityGroups", []) or []:
            gid = sg.get("GroupId")
            if gid:
                sgs.append(gid)

        # Tags: 轉成 {Key: Value}
        tags = {}
        for t in inst.get("Tags", []) or []:
            k, v = t.get("Key"), t.get("Value")
            if k:
                tags[k] = v


        ec2_instances.append({
            "instance_id": instance_id,
            "instance_type": instance_type,
            "state": state,
            "public_ip": public_ip,
            "private_ip": private_ip,
            "availability_zone": az,
            "launch_time": launch_time,
            "ami_id": ami_id,
            "ami_name": ami_name,
            "security_groups": sgs,
            "tags": tags,
        })

    return ec2_instances


def _s3_normalize_region(loc):
    # get_bucket_location 的 LocationConstraint 在 us-east-1 可能是 None 或 "us-east-1"/"US"
    if not loc or str(loc).upper() == "US":
        return "us-east-1"
    return loc

def collect_s3_buckets(session):

    s3 = session.client("s3")

    # 1) 列出所有 buckets（global）
    resp = s3.list_buckets()
    buckets = resp.get("Buckets", [])

    s3_buckets = []

    for b in buckets:
        name = b.get("Name")
        creation_date = _iso(b.get("CreationDate"))

        # 2) 查 bucket 所在 region（可能為 None → us-east-1）
        loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
        region = _s3_normalize_region(loc)


        # 3) 以 bucket 所在 region 建立對應端點的 s3 client（避免跨區清單失敗/轉址）
        s3_regional = session.client("s3", region_name=region) if region else s3

        # 4) 以 paginator 粗估 object_count & size_bytes
        total_count = 0
        total_bytes = 0
        try:
            paginator = s3_regional.get_paginator("list_objects_v2")
            # 不指定 Prefix，統計整個 bucket；注意：大量物件時會花時間
            for page in paginator.paginate(Bucket=name):
                for obj in page.get("Contents", []) or []:
                    total_count += 1
                    total_bytes += int(obj.get("Size", 0))
        
        except s3_regional.exceptions.NoSuchBucket:
            # 列表過程中 bucket 被刪除的罕見情況
            pass
        except Exception:
            # 沒有列舉物件權限（只給了 ListAllMyBuckets）時，保留 None
            total_count = None
            total_bytes = None

        s3_buckets.append({
            "bucket_name": name,
            "creation_date": creation_date,
            "region": region,
            "object_count": total_count,     # 可能為 None（無權或錯誤）
            "size_bytes": total_bytes        # 可能為 None（無權或錯誤）
        })

    return s3_buckets



def _fmt_protocol(proto):
    return "all" if proto == "-1" else proto  # -1 代表所有協定

def _fmt_port_range(rule):
    proto = rule.get("IpProtocol")
    if proto == "-1":
        return "all"
    from_p = rule.get("FromPort")
    to_p   = rule.get("ToPort")
    if from_p is None or to_p is None:
        # 像 ICMP 可能沒有埠
        return "N/A"
    if from_p == to_p:
        return f"{from_p}-{to_p}"
    return f"{from_p}-{to_p}"

def _collect_peers(rule, direction="in"):
    """
    將一條 IpPermissions / IpPermissionsEgress 的來源/目的整理成字串清單。
    direction: "in" → source；"out" → destination
    """
    peers = []

    # IPv4
    for r in rule.get("IpRanges", []) or []:
        cidr = r.get("CidrIp")
        if cidr:
            peers.append(cidr)

    # IPv6
    for r in rule.get("Ipv6Ranges", []) or []:
        cidr6 = r.get("CidrIpv6")
        if cidr6:
            peers.append(cidr6)

    # Prefix List（VPC 端點等）
    for r in rule.get("PrefixListIds", []) or []:
        pl = r.get("PrefixListId")
        if pl:
            peers.append(f"prefix-list:{pl}")

    # 參照其他 SG
    for r in rule.get("UserIdGroupPairs", []) or []:
        sgid = r.get("GroupId")
        if sgid:
            peers.append(f"sg:{sgid}")

    # 若沒有任何來源/目的，AWS 預設 outbound 開放全部的情況，通常會有 0.0.0.0/0
    return peers if peers else (["(none)"] if direction == "in" else ["(none)"])

def _flatten_rules(ip_permissions, direction="in"):
    """將多條規則展平成 [{protocol, port_range, source/destination}, ...]"""
    rows = []
    for perm in ip_permissions or []:
        proto = _fmt_protocol(perm.get("IpProtocol"))
        port_range = _fmt_port_range(perm)
        peers = _collect_peers(perm, direction=direction)
        for peer in peers:
            rows.append({
                "protocol": proto,
                "port_range": port_range,
                "source" if direction == "in" else "destination": peer
            })
    return rows

def collect_security_groups(session):

    ec2 = session.client("ec2")
    
    security_groups = []

    paginator = ec2.get_paginator("describe_security_groups")
    for page in paginator.paginate():
        for sg in page.get("SecurityGroups", []) or []:
            group_id   = sg.get("GroupId")
            group_name = sg.get("GroupName")
            desc       = sg.get("Description")
            vpc_id     = sg.get("VpcId")

            inbound  = _flatten_rules(sg.get("IpPermissions"), direction="in")
            outbound = _flatten_rules(sg.get("IpPermissionsEgress"), direction="out")

            security_groups.append({
                "group_id": group_id,
                "group_name": group_name,
                "description": desc,
                "vpc_id": vpc_id,
                "inbound_rules": inbound,
                "outbound_rules": outbound
            })

    return security_groups


# ---- utils for table formatting ----
def _nz(x, dash="–"):
    return dash if x in (None, "", []) else x

def _iso_date_only(iso_ts):
    # "2025-09-16T14:30:00Z" -> "2025-09-16"
    if not iso_ts or "T" not in iso_ts: 
        return _nz(iso_ts)
    return iso_ts.split("T", 1)[0]

def _fmt_mb(bytes_val):
    if bytes_val in (None,):
        return "~"
    return f"~{bytes_val/1024/1024:.1f}"

def render_table(session, iam_users, ec2_instances, s3_buckets, sec_groups):
    lines = []


    sts = session.client("sts")
    identity = sts.get_caller_identity() 

    # Header
    acct = identity.get("Account")
    reg  = session.region_name or "(n/a)"
    tstr = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    tstr = tstr.replace("T"," ").replace("Z","")[:16] if tstr else _nz(None)
    lines.append(f"AWS Account: {acct} ({reg})")
    lines.append(f"Scan Time: {tstr} UTC")
    lines.append("")

    # IAM
    lines.append(f"IAM USERS ({len(iam_users)} total)")
    lines.append(f"{'Username':20} {'Create Date':12} {'Last Activity':14} {'Policies':8}")
    for u in iam_users:
        uname = u.get("username","")
        cdate = _iso_date_only(u.get("create_date"))
        last  = _iso_date_only(u.get("last_activity"))
        pol_n = len(u.get("attached_policies", []))
        lines.append(f"{uname[:20]:20} {cdate:12} {last:14} {pol_n:>8}")
    lines.append("")

    # EC2
    running = sum(1 for i in ec2_instances if i.get("state") == "running")
    stopped = sum(1 for i in ec2_instances if i.get("state") == "stopped")
    lines.append(f"EC2 INSTANCES ({running} running, {stopped} stopped)")
    lines.append(f"{'Instance ID':20} {'Type':10} {'State':10} {'Public IP':16} {'Launch Time':19}")
    for i in ec2_instances:
        iid   = i.get("instance_id","")
        itype = i.get("instance_type","")
        st    = i.get("state","")
        pip   = _nz(i.get("public_ip"))
        ltime = i.get("launch_time","")
        # 顯示到分鐘
        ltime_short = ltime.replace("T"," ").replace("Z","")[:16] if ltime else _nz(None)
        lines.append(f"{iid:20} {itype:10} {st:10} {pip:16} {ltime_short:19}")
    lines.append("")

    # S3
    lines.append(f"S3 BUCKETS ({len(s3_buckets)} total)")
    lines.append(f"{'Bucket Name':28} {'Region':12} {'Created':12} {'Objects':8} {'Size (MB)':10}")
    for b in s3_buckets:
        name  = b.get("bucket_name","")
        rgn   = _nz(b.get("region"))
        cdate = _iso_date_only(b.get("creation_date"))
        cnt   = _nz(b.get("object_count"))
        szmb  = _fmt_mb(b.get("size_bytes"))
        lines.append(f"{name[:28]:28} {rgn:12} {cdate:12} {str(cnt):>8} {szmb:>10}")
    lines.append("")

    # Security Groups
    lines.append(f"SECURITY GROUPS ({len(sec_groups)} total)")
    lines.append(f"{'Group ID':16} {'Name':16} {'VPC ID':14} {'Inbound Rules':14}")
    for g in sec_groups:
        gid   = g.get("group_id","")
        gname = g.get("group_name","")
        vpc   = _nz(g.get("vpc_id"))
        in_n  = len(g.get("inbound_rules", []))
        lines.append(f"{gid:16} {gname[:16]:16} {vpc:14} {in_n:>14}")
    lines.append("")

    return "\n".join(lines)


def collect_account_info(session, region):
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    return {
        "account_id": identity.get("Account"),
        "user_arn": identity.get("Arn"),
        "region": region or session.region_name,
        "scan_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }


def main():

    args = parse_args()

    # region = get_region(args)
    # session = get_session(region)


    try:
        session = boto3.Session()

        if args.region and session.region_name != args.region:
            print("Region invalid.")
            sys.exit(1)


        iam_users = collect_iam_users(session)
        ec2_instances = collect_ec2_instances(session)
        s3_buckets = collect_s3_buckets(session)
        security_groups = collect_security_groups(session)
        # print(security_groups)

    except NoCredentialsError:
        print("No AWS credentials found. Set env vars or run 'aws configure'.")
        sys.exit(1)
    except NoRegionError:
        print("No region specified. Use --region or set AWS_DEFAULT_REGION.")
        sys.exit(1)
    except EndpointConnectionError as e:
        print(f"Endpoint/region error: {e}")
        sys.exit(1)
    except ClientError as e:
        print(f"Authentication failed (STS): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected auth error: {e}")
        sys.exit(1)




    if args.format == "json":

        sts = session.client("sts")
        identity = sts.get_caller_identity()  

        json_output = {
            "account_info": {
                "account_id": identity["Account"],
                "user_arn": identity["Arn"],
                "region": session.region_name,
                "scan_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            },
            "resources": {
                "iam_users": iam_users,
                "ec2_instances": ec2_instances,
                "s3_buckets": s3_buckets,
                "security_groups": security_groups
            },
            "summary": {
                "total_users": len(iam_users),
                "running_instances": len(ec2_instances),
                "total_buckets": len(s3_buckets),
                "security_groups": len(security_groups)
            }
        }

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json.dumps(json_output, indent=2))
            print(f"Saved json to {args.output}")
        else:
            print(json_output)

    else: # table

        table_output = render_table(session, iam_users, ec2_instances, s3_buckets, security_groups)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(table_output)
            print(f"Saved table to {args.output}")
        else:
            print(table_output)

    


   


if __name__ == "__main__":
    main()
