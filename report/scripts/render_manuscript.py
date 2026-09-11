"""Small deterministic converter for this report's deliberately limited Markdown.
Math and explicit LaTeX references pass through unchanged; prose escapes %, &.
Edit draft/*.md, not generated sections/*.tex. No network or model calls.
"""
from pathlib import Path
import csv
import re
R=Path(__file__).resolve().parents[1]
(R/'sections').mkdir(exist_ok=True)
def exact_wide(kind,body,label,caption):
    """Place a wide band between balanced IEEE-width text columns."""
    caption_line='\\captionof{'+kind+'}{'+caption+'}\\label{'+label+'}'
    content=(caption_line+'\n'+body) if kind=='table' else (body+'\n'+caption_line)
    return ('\\stopcolumns\n\\noindent\\begin{minipage}{\\textwidth}\n\\centering\n'+content+'\n'
            '\\end{minipage}\n\\par\\startcolumns\n')
def figure(name,label,caption,width='\\textwidth',wide=True,options=None):
    if name in {'prediction_response','representation_response','confidence_representation_scatter','class_heatmap'}:
        name += '_compact'
    graphic_options=options or 'width='+width
    body='\\includegraphics['+graphic_options+']{figures/'+name+'.pdf}'
    if wide:
        return exact_wide('figure',body,label,caption)
    return ('\\begin{figure}[H]\n\\centering\n'+body+'\n'
            '\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure}\n')
def table(name,label,caption,wide=False):
    body='\\input{tables/'+name+'.tex}'
    if wide:
        return exact_wide('table',body,label,caption)
    return ('\\begin{table}[H]\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n'
            +body+'\n\\end{table}\n')
assets={
 'framework':figure('framework','fig:framework','Five-lane experimental workflow and partition roles. The 27,000 EuroSAT RGB images are divided by content group into training (16,200), validation (4,050), calibration (2,700), and final test (4,050). Validation selects checkpoints and fits model-specific control ECDFs; calibration fits nested failure detectors. The protocol is frozen before nine transformed doses and three identity shams are evaluated on each final-test source. Paired predictions and representations support RQ1, RQ2, and RQ3. No human semantic-preservation review was performed.',options='width=\\textwidth,height=.82\\textheight,keepaspectratio'),
 'split':table('split','tab:split','Four-way split by image and content group. Groups are defined by exact and near-duplicate content, without geographical separation.'),
 'clean':table('clean','tab:clean','Clean final-test performance, mean $\\pm$ sample SD across three seeds. Accuracy and macro-F1 are percentages; ECE, Brier and NLL use their native scales. Each run evaluates 4,050 sources. Bold marks the better model for each metric; higher is better for accuracy and macro-F1, while lower is better for ECE, Brier and NLL.'),
 'contrasts':table('contrasts','tab:contrasts','Primary paired severe-minus-mild effects, mean $\\pm$ sample SD across seeds. Positive values indicate increasing instability. JS uses natural logarithms; ECDF is on a 0--1 percentile scale. Each per-seed contrast uses the same 4,050 sources. Within each family, bold marks the lower sensitivity value; displayed ties are bold for both models.',True),
 'detectors':table('detectors','tab:detectors','Failure-discrimination scores and paired gains, mean $\\pm$ sample SD across three seeds. AP denotes average precision. Populations and prevalence differ by model/seed (Appendix~\\ref{app:detector}); paired gains compare detectors on the same rows within each run. Within each model, bold marks the highest AP or AUROC among the five absolute score variants.',True),
 'prediction':figure('prediction_response','fig:prediction','Prediction response across the three frozen doses of each family. Points show means across three seeds, and error bars show sample SD. Each condition uses 4,050 sources per run. Lines connect the evaluated doses; intermediate severities were not measured.'),
 'representation':figure('representation_response','fig:representation','Raw within-model cosine (top) and validation-control ECDF instability (bottom), shown as mean and sample SD across seeds. The bottom panels use different scales because RGB gain and amplitude mixing occupy the ECDF lower tail. Near-zero percentiles can coexist with changed predictions. Larger ECDF values indicate greater displacement relative to the run\'s controls.'),
 'detector_figure':figure('detector_deltas','fig:detectors','Incremental detector performance by training seed. Points show paired gains from adding representation to the two output features. Bars are stored 95\\% source-group bootstrap CIs from 2,000 replicates, conditional on each trained model and fitted detector. Every interval excludes zero; coverage is individual rather than simultaneous.'),
 'scatter':figure('confidence_representation_scatter','fig:scatter','Confidence drop and representation instability for all non-sham pairs: 109,350 observations per model (4,050 sources, nine conditions, three seeds). Positive horizontal values indicate a confidence decrease; negative values indicate an increase. The vertical coordinate is a validation-control percentile, not a failure probability. Both panels share scales and include clean errors. Sources recur across seeds and conditions, so observations are dependent. No correlation test is attached to this display.'),
 'classes':figure('class_heatmap','fig:classes','Severe-condition label consistency by class, averaged across three seeds. Each cell includes all class sources, including clean errors. Consistency measures agreement with the clean prediction; accuracy would compare with the dataset label. The shared scale runs from zero (purple) to one (yellow), with rounded values printed in each cell. Class sample counts appear in Appendix~\\ref{app:counts}. Cell differences are descriptive and have no class-wise significance tests.'),
 'models':r'''\stopcolumns
\noindent
\begin{minipage}{\textwidth}
\centering
\captionof{table}{Model variants and training settings. Parameter counts include the ten-class head. Differences in pretraining and hardware limit attribution of performance gaps to architecture.}\label{tab:models}
\begin{tabular}{lll}\toprule
Field & ResNet-50 & ViT-Small/16 \\\midrule
timm variant & \texttt{resnet50.tv\_in1k} & \texttt{vit\_small\_patch16\_224.augreg\_in1k} \\
Trainable parameters & 23,528,522 & 21,669,514 \\
Representation & Pre-logit global average pool & Final normalized CLS token \\
Representation dimensions & 2,048 & 384 \\
Input / pretraining & $224\times224$ RGB / ImageNet-1k & $224\times224$ RGB / ImageNet-1k \\
Learning rate & $3\times10^{-4}$ & $5\times10^{-4}$ \\
Training device & H100 80GB & A100 MIG 3g.20gb \\
Training seeds & 17, 29, 43 & 17, 29, 43 \\\bottomrule
\end{tabular}
\end{minipage}
\par\startcolumns''',
 'deviations':r'''\stopcolumns
\noindent
\begin{minipage}{\textwidth}
\centering
\captionof{table}{Differences between the proposal or earlier template and the executed study.}\label{tab:deviations}
\begin{tabular}{p{.19\textwidth}p{.31\textwidth}p{.41\textwidth}}\toprule
Decision & Earlier plan/template & Executed study reported here \\\midrule
Models & ResNet-18 and compact DeiT/ViT variants; broader operational model matrix & ResNet-50 and ViT-Small/16; seeds 17, 29, 43 \\
Partition roles & Earlier split descriptions & Four fixed partitions, with separate detector calibration and final test \\
Semantic review & Planned human acceptance/adjudication & No human semantic review; all predeclared conditions retained by design \\
Fourier operator & Earlier masked-amplitude descriptions & Full-spectrum convex amplitude mixing, same-class training donor \\
Optional extensions & Spatial, MAE, multispectral and gamma directions & Not executed; no corresponding empirical claims \\
Report generation & Automated final artifact generation & Vocabulary repair; later report-index provenance defect independently audited \\\bottomrule
\end{tabular}
\end{minipage}
\par\startcolumns'''
}
def inline(s):
    s=s.replace('%',r'\%').replace('&',r'\&')
    s=re.sub(r'`([^`]+)`',lambda m:r'\path{'+m[1]+'}',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'\\textbf{\1}',s)
    s=re.sub(r'\[@([^\]]+)\]',lambda m:r'\cite{'+','.join(x.strip().lstrip('@') for x in m[1].split(';'))+'}',s)
    s=s.replace(' ± ',r' $\pm$ ').replace('---','---').replace('–','--').replace('−','-')
    return s
def render(text):
    if '{{clean}}' in text:
        before,after=text.split('{{clean}}',1)
        paragraphs=after.strip().split('\n\n')
        # Keep the one-column table beside its immediate interpretation. This
        # avoids a tall indivisible table unbalancing the surrounding columns.
        explanation='\n\n'.join(inline(p) for p in paragraphs[:2])
        assets['clean_pair']=(r'\stopcolumns'+'\n'+
            r'\noindent\begin{minipage}[t]{\dimexpr(\textwidth-\columnsep)/2\relax}'+'\n'+
            assets['clean']+'\n'+r'\end{minipage}\hfill'+
            r'\begin{minipage}[t]{\dimexpr(\textwidth-\columnsep)/2\relax}'+'\n'+
            explanation+'\n'+r'\end{minipage}\par\startcolumns')
        text=before+'{{clean_pair}}\n\n'+'\n\n'.join(paragraphs[2:])+'\n'
    out=[]; raw=False; bullets=False
    for line in text.splitlines():
        if line.startswith('{{'):
            out.append(assets[line[2:-2]]); continue
        if line==r'\[': raw=True
        if raw:
            out.append(line)
            if line==r'\]':raw=False
            continue
        if line.startswith('- '):
            if not bullets:out.append(r'\begin{itemize}');bullets=True
            out.append(r'\item '+inline(line[2:]));continue
        if bullets and line.strip():out.append(r'\end{itemize}');bullets=False
        m=re.match(r'^(#{1,3}) (.*?)(?: \{#([^}]+)\})?$',line)
        if m:
            cmd=['section','subsection','subsubsection'][len(m[1])-1]
            out.append('\\'+cmd+'{'+inline(m[2])+'}'+(r'\label{'+m[3]+'}' if m[3] else ''))
        else:out.append(inline(line))
    if bullets:out.append(r'\end{itemize}')
    return '\n'.join(out)+'\n'
# Transpose the presentation only; retain the canonical CSV and all precision.
with (R/'tables/clean.csv').open(newline='',encoding='utf-8') as stream:
    clean_rows=list(csv.reader(stream))
clean_lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Metric & ResNet-50 & ViT-Small/16 \\',r'\midrule']
for index,metric in enumerate(clean_rows[0][1:],1):
    clean_lines.append(metric+' & '+r'\textbf{'+clean_rows[1][index]+'} & '+clean_rows[2][index]+r' \\')
clean_lines.extend([r'\bottomrule',r'\end{tabular}'])
(R/'tables/clean.tex').write_text('\n'.join(clean_lines)+'\n',encoding='utf-8')
with (R/'tables/detectors.csv').open(newline='',encoding='utf-8') as stream:
    detector_rows=list(csv.DictReader(stream))
detector_lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{ResNet-50} & \multicolumn{2}{c}{ViT-Small/16} \\',r'Score / paired gain & AP & AUROC & AP & AUROC \\',r'\midrule']
for score in dict.fromkeys(row['Score / paired gain'] for row in detector_rows):
    cells=[score]
    for model in ['ResNet-50','ViT-Small/16']:
        row=next(row for row in detector_rows if row['Model']==model and row['Score / paired gain']==score)
        for key in ['AP','AUROC']:
            value=row[key]
            cells.append(r'\textbf{'+value+'}' if score=='Output + representation' else value)
    detector_lines.append(' & '.join(cells)+r' \\')
detector_lines.extend([r'\bottomrule',r'\end{tabular}'])
(R/'tables/detectors.tex').write_text('\n'.join(detector_lines)+'\n',encoding='utf-8')
for p in sorted((R/'draft').glob('*.md')):
    (R/'sections'/(p.stem+'.tex')).write_text(render(p.read_text(encoding='utf-8')),encoding='utf-8')
print('MANUSCRIPT_RENDERED')
