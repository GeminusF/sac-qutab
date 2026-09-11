"""Measure the assembled layout without claiming automatic visual approval."""
from pathlib import Path
import hashlib,json,re,subprocess,unicodedata
R=Path(__file__).resolve().parents[1]
pdf=R/'report.pdf'
def normalize(text):
    return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKC',text).lower())
# Raw content order preserves the separate multicols bands. Default Poppler
# reconstruction interleaves left/right lines on page 8; visually verified.
text=subprocess.check_output(['pdftotext','-raw',str(pdf),'-'],encoding='utf-8')
words=normalize(text)
coverage=[]
for path in sorted((R/'draft').glob('*.md')):
    for paragraph in path.read_text(encoding='utf-8').split('\n\n'):
        if not paragraph or paragraph.startswith(('#','{{','\\','- ')):continue
        # A prefix ending before math/citation markup checks prose survived
        # composition across both columns and pages.
        prefix=re.split(r'\\|\[|\*|`|\$',paragraph)[0].strip()
        prefix=' '.join(prefix.split()[:12])
        if len(prefix)<35:continue
        coverage.append({'source':path.name,'prefix':prefix,'present':normalize(prefix) in words})
aux=(R/'report.aux').read_text(encoding='utf-8')
labels={m[1]:{'number':m[2],'page':int(m[3])} for m in re.finditer(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',aux)}
main=int(re.search(r'\\newlabel\{LastMainPage\}\{\{[^}]*\}\{(\d+)\}',aux)[1])
pagecount=int(re.search(r'Pages:\s+(\d+)',subprocess.check_output(['pdfinfo',str(pdf)],encoding='utf-8'))[1])
positions=[]
for path in sorted((R/'draft').glob('*.md')):
    paragraphs=path.read_text(encoding='utf-8').split('\n\n')
    for i,p in enumerate(paragraphs):
        if p.startswith('{{'):
            positions.append({'asset':p.strip(),'source':path.name,'introducing_paragraph':paragraphs[i-1]})
report={'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'total_pages':pagecount,
        'main_pages':main,'appendix_pages':pagecount-main,'labels':labels,'source_anchors':positions,
        'prose_prefix_coverage':coverage,'missing_prose_prefixes':[r for r in coverage if not r['present']],
        'unresolved_reference_markers':'??' in text,'original_framework_preserved':hashlib.sha256((R/'figures/framework.pdf').read_bytes()).hexdigest()=='fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71'}
(R/'audit/layout_measurements.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in {'prose_prefix_coverage','labels','source_anchors'}},indent=2))
