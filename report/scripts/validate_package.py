import os
"""Report source QA plus verification of the separately recorded PDF review.

Requires only the Python standard library. It does not compile LaTeX or run the
scientific experiment; use audit_sources.py for the latter. PDF creation and
visual inspection are performed separately, then authenticated here.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys

R=Path(__file__).resolve().parents[1]
A=Path(os.environ.get('SAC_EVIDENCE_ROOT', str(R.parent/'all_artifacts')))
checks=[]
def check(ok,message):
    if not ok: raise AssertionError(message)
    checks.append(message)
def read(name): return json.loads((R/'audit'/name).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

for p in (R/'scripts').glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
check(True,'Python source syntax')

# Regeneration must not change already delivered sections.
before={p.name:p.read_bytes() for p in (R/'sections').glob('*.tex')}
subprocess.run([sys.executable,str(R/'scripts/render_manuscript.py')],check=True)
after={p.name:p.read_bytes() for p in (R/'sections').glob('*.tex')}
check(before==after,'Draft and generated LaTeX agree')

tex_files=[R/'report.tex',R/'appendix.tex',R/'references.tex',*sorted((R/'sections').glob('*.tex')),*sorted((R/'tables').glob('*.tex'))]
texts=[]
for p in tex_files:
    s=p.read_text(encoding='utf-8')
    s=re.sub(r'(?<!\\)%[^\n]*','',s)
    texts.append(s)
    depth=0
    for match in re.finditer(r'(?<!\\)[{}]',s):
        depth+=1 if match[0]=='{' else -1
        check(depth>=0,f'No premature closing brace: {p.relative_to(R)}') if depth<0 else None
    check(depth==0,f'Balanced braces: {p.relative_to(R)}')
    stack=[]
    for match in re.finditer(r'\\(begin|end)\{([^}]+)\}',s):
        if match[1]=='begin':stack.append(match[2])
        else:
            check(bool(stack) and stack.pop()==match[2],f'Environment nesting: {p.relative_to(R)} / {match[2]}')
    check(not stack,f'Closed environments: {p.relative_to(R)}')
alltext='\n'.join(texts)
for match in re.finditer(r'\\input\{([^}]+)\}',alltext):
    path=R/match[1]
    if not path.suffix:path=path.with_suffix('.tex')
    check(path.is_file(),f'Input exists: {match[1]}')
for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',alltext):
    check((R/match[1]).is_file(),f'Figure exists: {match[1]}')
labels=re.findall(r'\\label\{([^}]+)\}',alltext)
check(len(labels)==len(set(labels)),'Unique LaTeX labels')
for ref in re.findall(r'\\(?:ref|pageref)\{([^}]+)\}',alltext):check(ref in labels,f'Reference target: {ref}')
bib=re.findall(r'\\bibitem\{([^}]+)\}',alltext)
check(len(bib)==16 and len(set(bib))==16,'Sixteen distinct bibliography entries')
for citation in re.findall(r'\\cite\{([^}]+)\}',alltext):
    for key in citation.split(','):check(key in bib,f'Citation target: {key}')
check(not re.search(r'\{\{\w+\}\}|\[@|TODO|TBD|PLACEHOLDER',alltext),'No unresolved draft tokens or placeholders')
abstract=(R/'draft/00_abstract.md').read_text(encoding='utf-8')
check(200<=len(abstract.split())<=250,'Abstract within 200–250 whitespace-delimited words')
report_tex=(R/'report.tex').read_text(encoding='utf-8')
for package in ['float','capt-of','multicol']:
    check(f'\\usepackage{{{package}}}' in report_tex,f'Global exact placement loads {package}')
check('\\usepackage{cuted}' not in report_tex,'Unreliable strip output path is disabled')
check(not re.search(r'\\begin\{(?:figure|table)\*\}',alltext),'No two-column float environments remain')
ordinary_visuals=re.findall(r'\\begin\{(figure|table)\}(?:\[([^]]*)\])?',alltext)
check(len(ordinary_visuals)==6 and all(spec=='H' for _,spec in ordinary_visuals),'Every ordinary figure and table uses H placement')
check('\\begin{strip}' not in alltext and '\\end{strip}' not in alltext,'No strip environments remain')
check('\\twocolumn[{' not in alltext,'No forced page-top wide visual blocks')
section_text='\n'.join(p.read_text(encoding='utf-8') for p in (R/'sections').glob('*.tex'))
check('\\clearpage' not in section_text,'No per-visual forced page breaks')
check(section_text.count('\\stopcolumns')==11 and section_text.count('\\startcolumns')==11,'Ten wide visuals and the paired clean table preserve column-band text flow')
check(alltext.count('\\captionof{figure}')==6 and alltext.count('\\captionof{table}')==4,'Full-width captions retain figure and table types')
check('width=\\textwidth,height=.82\\textheight,keepaspectratio' in section_text,'Figure 1 preserves its approved placement dimensions; font limit is separately disclosed')
approved_framework_sha='fb052ea348c38f35c014df85d24e93bbbc305d71101ec52a83752757ac257b71'
framework_source=R/'figures/framework.pdf'
check(framework_source.is_file() and digest(framework_source)==approved_framework_sha,'Approved Figure 1 source hash')
check(digest(R/'figures/framework.pdf')==approved_framework_sha,'Canonical Figure 1 matches approved source')
asset_builder=(R/'scripts/build_assets.py').read_text(encoding='utf-8')
check("save(fig,'framework')" not in asset_builder and 'FancyBboxPatch' not in asset_builder,'Asset rebuild cannot redraw the approved Figure 1')
new_title='Semantic Analysis of Counterfactuals: Quality Under Transformations and Adaptive Benchmarking'
old_title='SAC-QUTAB: Paired Prediction and Representation Sensitivity on EuroSAT RGB'
check(report_tex.count(new_title)==1 and f'pdftitle={{{new_title}}}' in report_tex,'Visible title and PDF title metadata agree')
check(old_title not in alltext,'Superseded title absent from current report source')
check('Semantic Analysis of Counterfactuals: Quality Under Transformations and Adaptive Benchmarking (SAC-QUTAB)' in abstract,'Acronym expanded in Abstract')
check('Semantic Analysis of Counterfactuals: Quality Under Transformations and Adaptive Benchmarking (SAC-QUTAB)' in (R/'draft/01_introduction.md').read_text(encoding='utf-8'),'Acronym expanded in Introduction')
authors=['Milana Karimova','Sharaf Feyzullayev','Farah Feyzullayev','Jeyhuna Sevdiyeva','Nicat Aghayev']
author_positions=[report_tex.index(name) for name in authors]
check(author_positions==sorted(author_positions),'Contribution-based author order')
check('pdfauthor={'+', '.join(authors)+'}' in report_tex,'PDF author metadata agrees with visible order')
check('Aghalarov' not in alltext and 'Nicat Aghayev' in report_tex,'Nicat Aghayev spelling corrected')
emails=['milana.karimova25@aiacademy.edu.az','sheref.feyzullayev25@aiacademy.edu.az','fereh.feyzullayev25@aiacademy.edu.az','ceyhune.sevdiyeva25@aiacademy.edu.az','nicat.agayev25@aiacademy.edu.az']
for name,email in zip(authors,emails):
    check(f'\\sacauthor{{{name}}}{{{email}}}' in report_tex,f'Author and email paired: {name}')
check('\\IEEEauthorrefmark' not in report_tex,'No author-reference marks')
check(report_tex.count('National Intelligence Center of Azerbaijan')==1 and report_tex.count('\\sacauthor{')==5,'Five author blocks share the required two-line affiliation template')
check('AI Academy, Azerbaijan' not in report_tex,'No redundant shared affiliation line')
check('[@repository]' in abstract and 'github.com' not in abstract.lower(),'Abstract cites repository without exposing URL')
keywords=re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}',report_tex,re.S).group(1)
check(not re.search(r'—|---|\\textemdash|\\emdash',abstract+keywords),'No em dash in Abstract or Index Terms')
check('Tectonic compiled' not in alltext and 'Poppler rendered' not in alltext,'No unperformed report compilation claim')

audit=read('source_audit.json')
check(len(audit['runs'])==6 and all(r['verified'] for r in audit['runs']),'Six saved run audit records passed')
check(len(audit['historical_report_index_anomaly'])==5,'Five historical index anomalies disclosed')
check(all(r['pair_count']==48600 and r['source_count']==4050 for r in audit['runs']),'Expected paired/source denominators')
clean=read('clean.json'); det=read('detectors.json'); cond=read('conditions.json')
check(len(clean)==6 and len(det)==30 and len(cond)==72,'Six clean / thirty detector / seventy-two condition summaries')
headlines={}
for model in ['resnet50','vit_small_patch16_224']:
    cc=[r['accuracy']*100 for r in clean if r['model']==model]
    severe=[r['transformed_accuracy']*100 for r in cond if r['model']==model and r['family']=='resolution' and r['severity']=='severe']
    gains=[]
    for seed in [17,29,43]:
        scores={r['score']:r for r in det if r['model']==model and r['seed']==seed}
        gains.append(scores['output_plus_representation']['AP']-scores['output_only']['AP'])
    headlines[model]={'clean_accuracy_pct':statistics.mean(cc),'clean_accuracy_sd_pp':statistics.stdev(cc),'severe_resolution_accuracy_pct':statistics.mean(severe),'AP_gain':statistics.mean(gains),'AP_gain_SD':statistics.stdev(gains)}
    for value,decimals in [(statistics.mean(cc),2),(statistics.mean(severe),2),(statistics.mean(gains),3)]:
        check(f'{value:.{decimals}f}' in abstract,f'Abstract number traced: {model} / {value:.{decimals}f}')
stats=read('primary_statistics.json')
intervals=[r for r in stats['primary_per_seed'] if r['contrast'].startswith('rq3_nested_detector_delta_')]
check(len(intervals)==12 and all(r['ci'][0]>0 for r in intervals),'All twelve stored paired detector CIs above zero')
expected_figures={'framework','prediction_response','representation_response','detector_deltas',
                   'class_heatmap','learning_curves','confidence_representation_scatter',
                   'qualitative_failure_gallery_1','qualitative_failure_gallery_2',
                   'attention_rollout_repaired'}
check(expected_figures <= {p.stem for p in (R/'figures').glob('*.png')},'Ten required figure PNGs')
check(len(list((R/'tables').glob('*.tex')))==9,'Nine generated numeric tables')
baseline=read('revision_baseline.json')
for name in ['class_counts','clean','contrasts','delta_ci','detectors','full_conditions','per_seed_clean','population','split']:
    check(digest(R/'tables'/f'{name}.csv')==baseline['report_files'][f'tables\\{name}.csv'],f'Numeric table source unchanged: {name}.csv')

table_text={name:(R/'tables'/f'{name}.tex').read_text(encoding='utf-8') for name in ['clean','contrasts','detectors','per_seed_clean','full_conditions','delta_ci']}
check(table_text['clean'].count('\\textbf{')==5,'Clean table bolds five direction-aware best cells')
check(table_text['contrasts'].count('\\textbf{')==10,'Contrast table bolds the lower-sensitivity cells and displayed tie')
check(table_text['detectors'].count('\\textbf{')==4,'Detector table bolds best absolute AP and AUROC within model')
check(table_text['per_seed_clean'].count('\\textbf{')==5 and '& 39 \\\\' in table_text['per_seed_clean'],'Per-seed clean table bolds only five best metric cells')
check(table_text['full_conditions'].count('\\textbf{')==38,'Full-condition table bolds model-wise best cells including displayed ECDF ties')
check(table_text['delta_ci'].count('\\textbf{')==4 and not re.search(r'\\textbf\{\[[^}]+\]\}',table_text['delta_ci']),'Gain table bolds four best gains without bolding CI limits')

archived=[]
for pattern,expected in [('*.csv',15),('*.png',47)]:
    paths=sorted((A/'report_artifacts/final').glob(pattern))
    check(len(paths)==expected,f'Original supplementary count {pattern}: {expected}')
    for src in paths:
        dst=R/'supplementary'/src.name
        check(dst.is_file() and digest(src)==digest(dst),f'Unmodified supplementary copy: {src.name}')
        archived.append({'file':src.name,'sha256':digest(dst)})
for required in ['README.md','audit/claim_evidence.md','audit/chart_map.md','audit/reference_audit.md','audit/qa_status.md','audit/presentation_revision_20260911.md']:
    check((R/required).is_file(),f'Required handoff file: {required}')

revision=read('visual_revision.json')
check(headlines==baseline['headlines'],'Headline calculations unchanged from pre-revision baseline')
for name,sha in baseline['original_visuals'].items():
    check(digest(A/'report_artifacts/final'/name)==sha,f'Original report artifact preserved: {name}')
check(revision['scatter_counts']=={'resnet50':109350,'vit_small_patch16_224':109350},'All non-sham scatter observations retained')
for row in revision['sources']:
    path=A/'evaluations'/row['run']/'final_test/pair_metrics.jsonl'
    check(digest(path)==row['pair_sha256'],'Visual source pairs unchanged: '+row['run'])
with (A/'report_artifacts/final/condition_class_metrics.csv').open(encoding='utf-8',newline='') as stream:
    class_rows=list(csv.DictReader(stream))
for i,cls in enumerate(revision['heatmap']['classes']):
    for j,(model,family) in enumerate((m,f) for m in ['resnet50','vit_small_patch16_224'] for f in revision['heatmap']['families']):
        rows=[r for r in class_rows if r['class_name']==cls and r['model']==model and r['family']==family and r['severity']=='severe']
        check(len(rows)==3 and {int(r['seed']) for r in rows}=={17,29,43},f'Heatmap includes all seeds: {cls}/{model}/{family}')
        check(abs(statistics.mean(float(r['label_consistency']) for r in rows)-revision['heatmap']['values'][i][j])<1e-12,f'Heatmap matches original class CSV: {cls}/{model}/{family}')
check(revision['gallery_selection_replayed'] and revision['gallery_inputs_hash_verified'],'Gallery selection replay and input hash checks passed')
check(revision['gallery_labels_verified'],'Gallery labels and predictions match saved probabilities')
for row in revision['gallery_rows']:
    check(digest(R/'audit/visual_inputs'/(row['source_id']+'.jpg'))==row['source_sha256'],'Gallery original image hash: '+row['source_id'])
    check(digest(R/'audit/visual_inputs'/(row['pair_id']+'.npy'))==row['cache_sha256'],'Gallery transformed cache hash: '+row['pair_id'])
logo_order=['AZCON_logo.png','naic_logo.png','academy_logo.png','sac_qutab_logo.png']
title=report_tex
check([r['file'] for r in revision['logos']]==logo_order,'Requested logo source order')
positions=[title.index('figures/logos/'+name) for name in logo_order]
check(positions==sorted(positions),'Requested logo order in title block')
rule_position=title.index(r'\rule{.96\textwidth}{0.4pt}')
visible_title_position=title.find('Semantic Analysis of Counterfactuals:',positions[3])
check(positions[2] < rule_position < positions[3] < visible_title_position,'Institutional logo bar, rule and SAC-QUTAB title row ordered correctly')
logo_source_drift=[]
canonical=read('canonical_assets.json')
for rel, expected in canonical.items():
    check(digest(R/rel)==expected,'Canonical asset hash: '+rel)
for row in revision['logos']:
    check(digest(R/'figures/logos'/row['file'])==row['trimmed_sha256'],'Trimmed logo hash: '+row['file'])
prose='\n'.join(p.read_text(encoding='utf-8') for p in sorted((R/'draft').glob('*.md')))+(R/'appendix.tex').read_text(encoding='utf-8')
check(not re.search(r'—|---|\\textemdash|\\emdash',prose),'No em dash or TeX em-dash sequence in authored prose')
authored_tex='\n'.join([report_tex,(R/'appendix.tex').read_text(encoding='utf-8'),*[(R/'sections'/(p.stem+'.tex')).read_text(encoding='utf-8') for p in sorted((R/'draft').glob('*.md'))]])
check(not re.search(r'—|---|\\textemdash|\\emdash',authored_tex),'Generated prose and captions contain no em dash')
# Sentence-initial transition fillers are distinct from content-bearing uses
# such as "Full experiment reproduction additionally requires ...".
patterns=[r'\bIt is important to note that\b',r'\b(?:Notably|Importantly|Interestingly|clearly|obviously)\b',
          r'(?:^|[.!?]\s+|\n\s*)(?:Furthermore|Moreover|Additionally)\b',
          r'\bdelve into\b',r'\bshed light on\b',r'\ba rich tapestry\b',r'\bThese findings (?:underscore|highlight)\b']
check(not any(re.search(p,prose,re.I) for p in patterns),'No selected formulaic prose patterns')
sentences=re.split(r'(?<=[.!?])\s+|\n\n',prose)
starts={word:sum(bool(re.match(word+r'\b',s)) for s in sentences) for word in ['It','This','We']}
check(not any(re.match(r'(It|This|We)\b',a) and re.match(r'(It|This|We)\b',b) for a,b in zip(sentences,sentences[1:])),'No adjacent It/This/We sentence openings')
attention=read('attention_repair_status.json')
check(attention['status']=='COMPLETED' and attention['inference_performed'] and attention['figure_visually_verified'],'Attention repair generated and visually reviewed')
check(attention['forward_calls']==1 and attention['input_images']==2 and attention['predictions']==['HerbaceousVegetation','PermanentCrop'],'Attention replay stayed within the approved pair')
check(attention['confidence_max_absolute_difference'] <= attention['confidence_tolerance'],'Attention replay confidence within declared tolerance')
check(digest(R/'figures/attention_rollout_repaired.png')==attention['figure_png_sha256'],'Attention PNG hash')
check(digest(R/'figures/attention_rollout_repaired.pdf')==attention['figure_pdf_sha256'],'Attention PDF hash')
check(digest(R/'audit/attention_rollout_raw.npz')==attention['raw_npz_sha256'],'Raw attention output hash')
check('fig:attention' in labels and 'No training, detector fitting or full-test inference was repeated' in alltext,'Attention scope and interpretation disclosed')

pdf_qa=read('pdf_qa.json')
pdf=R/pdf_qa['pdf']
check(pdf_qa['status']=='PASSED_WITH_DISCLOSED_LIMITS' and pdf_qa['compiled'],'Recorded PDF build passed with disclosed limits')
check(pdf.is_file() and digest(pdf)==pdf_qa['pdf_sha256'],'Compiled PDF hash matches QA record')
aux=(R/'report.aux').read_text(encoding='utf-8')
main_page=int(re.search(r'\\newlabel\{LastMainPage\}\{\{[^}]*\}\{(\d+)\}',aux)[1])
check(pdf_qa['main_pages_including_references']==main_page and pdf_qa['appendix_first_page']==main_page+1,'Main/appendix boundary agrees with compiled labels')
check(pdf_qa['total_pages']==main_page+pdf_qa['appendix_pages'],'Recorded page totals are consistent')
check(pdf_qa['visually_checked_pages']==list(range(1,pdf_qa['total_pages']+1)),'Every compiled PDF page visually checked')
check(not pdf_qa['unresolved_reference_markers'],'No unresolved reference markers in PDF text')
check(not pdf_qa['missing_or_unembedded_fonts'],'All PDF fonts embedded')
check(not pdf_qa['clipped_or_overlapping_content'] and not pdf_qa['blocking_layout_defects'],'No clipping, overlap or blocking placement defects')
check(not pdf_qa['minimum_figure_font_requirement_met'] and not pdf_qa['main_page_target_met'] and len(pdf_qa['disclosed_limits'])==2,'Figure 1 font floor and main-page target explicitly unresolved')
layout=read('layout_measurements.json')
check(layout['pdf_sha256']==digest(pdf) and not layout['missing_prose_prefixes'],'Rendered main-prose prefixes present in PDF content order')
result={'status':'SOURCE_AND_PDF_QA_PASSED_WITH_DISCLOSED_LIMITS','assembled_pdf_compiled':True,'assembled_pdf_visually_checked':True,'page_count_verified':True,'reason':'Source integrity and layout review completed; unchanged Figure 1 has sub-8pt source text, and the main-page target is reported separately','pdf_qa':pdf_qa,'logo_source_drift':logo_source_drift,'attention_repair':attention,'sentence_openings':starts,'check_count':len(checks),'checks':checks,'headlines':headlines,'supplementary_copy_hashes':archived}
(R/'audit/source_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(f"SOURCE_AND_PDF_QA_PASSED_WITH_DISCLOSED_LIMITS: {len(checks)} checks; {pdf_qa['total_pages']} pages; main report {main_page} pages. Openings: {starts}")
