"""Build references.bib from the arXiv API so titles/authors/years are exact."""
import re
import time
import urllib.request
import xml.etree.ElementTree as ET

KEYS = {
    "samvelyan2019smac": "1902.04043", "ellis2023smacv2": "2212.07489",
    "vinyals2017pysc2": "1708.04782", "smactalk2026": "2606.04202",
    "llmpysc2": "2411.05348", "ma2023textsc2": "2312.11865", "hlsmac2025": "2509.12927",
    "liu2025spiral": "2506.24119", "park2025maporl": "2502.18439", "liao2025marft": "2504.16129",
    "hasan2026grpo": "2606.07845", "strongermas2025": "2510.11062", "magrpo2025": "2508.04652",
    "schulman2017ppo": "1707.06347", "schulman2015gae": "1506.02438", "peng2019awr": "1910.00177",
    "yu2022mappo": "2103.01955", "rashid2018qmix": "1803.11485", "hu2020otherplay": "2003.02979",
    "hu2022lora": "2106.09685", "nakamura2026colosseum": "2602.15198",
    "shi2026emergent": "2609.24967", "motwani2024secret": "2402.07510",
    "zeng2026voluntary": "2605.27593", "piatti2024govsim": "2404.16698",
    "wei2022hok": "2209.08483", "agashe2023llmcoord": "2310.03903", "prost2025": "2509.04508",
    "ross2011dagger": "1011.0686", "kutasov2025shade": "2506.15740",
}
NS = {"a": "http://www.w3.org/2005/Atom"}
url = "http://export.arxiv.org/api/query?max_results=100&id_list=" + ",".join(KEYS.values())
xml = urllib.request.urlopen(url, timeout=60).read()
root = ET.fromstring(xml)
by_id = {}
for e in root.findall("a:entry", NS):
    aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[1])
    by_id[aid] = {
        "title": " ".join(e.find("a:title", NS).text.split()),
        "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
        "year": e.find("a:published", NS).text[:4],
    }
out = []
for key, aid in KEYS.items():
    d = by_id.get(aid)
    if not d:
        print("MISSING", key, aid)
        continue
    title = d["title"].replace("&", r"\&")
    out.append(f"@article{{{key},\n  title = {{{{{title}}}}},\n  author = {{{' and '.join(d['authors'])}}},\n"
               f"  journal = {{arXiv preprint arXiv:{aid}}},\n  year = {{{d['year']}}},\n"
               f"  url = {{https://arxiv.org/abs/{aid}}}\n}}\n")
out.append("""@misc{gemma4,
  title = {{Gemma 4 model card}},
  author = {{Google DeepMind}},
  year = {2026},
  url = {https://huggingface.co/google/gemma-4-E4B-it}
}
""")
open("references.bib", "w").write("\n".join(out))
print(f"wrote {len(out)} entries")
for key, aid in KEYS.items():
    d = by_id.get(aid)
    if d:
        print(f"{key}: {d['authors'][0]} et al. ({d['year']}) {d['title'][:80]}")
