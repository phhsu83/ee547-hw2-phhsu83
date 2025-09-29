from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import urlparse, parse_qs
import sys
import re
from datetime import datetime


# load ArXiv paper data
with open("sample_data/papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

with open("sample_data/corpus_analysis.json", "r", encoding="utf-8") as f:
    corpus_analysis = json.load(f)


# 
class ArxivHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/papers":
            self.handle_papers()

        elif path.startswith("/papers/"):
            self.handle_single_paper(path.split("/")[-1])

        elif path == "/search":
            self.handle_search(query)

        elif path == "/stats":
            self.handle_stats()

        else:
            self.send_error(404, "Invalid endpoints")


    def handle_papers(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            filtered = [{"arxiv_id": p["arxiv_id"], 
                        "title": p["title"], 
                        "authors": p["authors"], 
                        "categories": p["categories"]} for p in papers]

            self.wfile.write(json.dumps(filtered).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 200 OK"
            print(log_msg)

        except Exception as e:
            # 捕捉任何沒預期的錯誤 → 回 500
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server error",
                "message": str(e)  # 開發測試用，可以回錯誤訊息
            }).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 500"
            print(log_msg)


    def handle_single_paper(self, arxiv_id):
        try:
            paper = next((p for p in papers if p["arxiv_id"] == arxiv_id), None)
            if paper:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                filtered = {"arxiv_id": paper["arxiv_id"], 
                            "title": paper["title"], 
                            "authors": paper["authors"], 
                            "abstract": paper["abstract"],
                            "categories": paper["categories"],
                            "published": paper["published"],
                            "abstract_stats": {
                                "total_words": paper["abstract_stats"]["total_words"],
                                "unique_words": paper["abstract_stats"]["unique_words"],
                                "total_sentences": paper["abstract_stats"]["total_sentences"]
                            }
                            }

                self.wfile.write(json.dumps(filtered).encode())

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"[{now}] {self.command} {self.path} - 200 OK"
                print(log_msg)

            else:
                self.send_error(404, "Unknown paper IDs")

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"[{now}] {self.command} {self.path} - 404"
                print(log_msg)

        except Exception as e:
            # 捕捉任何沒預期的錯誤 → 回 500
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server error",
                "message": str(e)  # 開發測試用，可以回錯誤訊息
            }).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 500"
            print(log_msg)



    # 
    def _tokenize_query(self, q: str):
        q = (q or "").strip().lower()
        # 用正則把連續的空白、tab、換行都當成分隔符
        terms = [t for t in re.split(r"\s+", q) if t]
        return terms

    
    def _count_term(self, text: str, term: str) -> int:
        """用 \bword\b 做整詞比對（不分大小寫），回傳出現次數。"""
        if not text or not term:
            return 0
        # 每次都直接編譯 regex，不做快取
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        return len(pattern.findall(text))

    def _score_paper(self, paper: dict, terms: list[str]):
        """回傳 (是否通過AND條件, match_score, matches_in)"""
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""
        title_hits_total = 0
        abstract_hits_total = 0

        # AND：每個 term 至少出現一次於 (title or abstract)
        for term in terms:
            c_title = self._count_term(title, term)
            c_abs = self._count_term(abstract, term)
            if (c_title + c_abs) == 0:
                return (False, 0, [])  # AND 失敗
            title_hits_total += c_title
            abstract_hits_total += c_abs

        matches_in = []
        if title_hits_total > 0:
            matches_in.append("title")
        if abstract_hits_total > 0:
            matches_in.append("abstract")

        match_score = title_hits_total + abstract_hits_total
        return (True, match_score, matches_in)


    def handle_search(self, query):

        try:
            raw_q = query.get("q", [None])[0]
            terms = self._tokenize_query(raw_q)

            # 驗證：缺 q 或全部是空白 → 400
            if not terms:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                
                self.send_error(400, "Malformed search queries")

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_msg = f"[{now}] {self.command} {self.path} - 400"
                print(log_msg)
                return

            results = []
            for p in papers:
                ok, score, where = self._score_paper(p, terms)
                if not ok:
                    continue
                results.append({
                    "arxiv_id": p.get("arxiv_id"),
                    "title": p.get("title"),
                    "match_score": score,
                    "matches_in": where
                })

            # 依分數高到低排序；同分以 arxiv_id 當次序鍵（可依需求調整）
            results.sort(key=lambda r: (-r["match_score"], str(r["arxiv_id"])))

            resp = {
                "query": raw_q,
                "results": results
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 200 OK"
            print(log_msg)

        except Exception as e:
            # 捕捉任何沒預期的錯誤 → 回 500
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server error",
                "message": str(e)  # 開發測試用，可以回錯誤訊息
            }).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 500"
            print(log_msg)


    def handle_stats(self):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            filtered = {"total_papers": corpus_analysis["corpus_stats"]["total_abstracts"], 
                        "total_words": corpus_analysis["corpus_stats"]["total_words"], 
                        "unique_words": corpus_analysis["corpus_stats"]["unique_words_global"], 
                        "top_10_words": [corpus_analysis["top_50_words"][i] for i in range(10)],
                        "category_distribution": corpus_analysis["category_distribution"]
                        }

            self.wfile.write(json.dumps(filtered).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 200 OK"
            print(log_msg)

        except Exception as e:
            # 捕捉任何沒預期的錯誤 → 回 500
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Server error",
                "message": str(e)  # 開發測試用，可以回錯誤訊息
            }).encode())

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_msg = f"[{now}] {self.command} {self.path} - 500"
            print(log_msg)


def main():


    

    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("0.0.0.0", port), ArxivHandler)
    print(f"Server running at http://localhost:{port}")
    server.serve_forever()





if __name__ == "__main__":
    main()