"""Render the proposed publication assignment; never invent historical work."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
ROOT=Path(__file__).resolve().parents[1]
ASSIGNMENTS=[
 ['Farah Feyzullayev','GeminusF','Package integration; PR/review templates','Reproduction entry points and release references','Training runbook and freeze/resume documentation','Review narrative consistency','PRs 1, 6, 11; reviews 5, 9, 13'],
 ['Nicat Aghayev','Nicat-Agayev','Resource validation; provenance regression; contribution tools','Data, weights and provenance documentation','Verify saved-output provenance','Review experiment explanations','PRs 2, 7, 12; reviews 1, 10, 14'],
 ['Sharaf Feyzullayev','optim00s','Environment profiles; report build; CI checks','Report source and canonical assets','Check runtime compatibility','Review model and setup material','PRs 3, 8, 13; reviews 2, 6, 15'],
 ['Milana Karimova','milanakarimova','Evidence packaging; report audit','Claim/evidence mapping; final manuscript QA','Verify evidence hashes and release inputs','Review results and limitations','PRs 4, 9, 14; reviews 3, 7, 11'],
 ['Jeyhuna Sevdiyeva','Jeyhunaa','Headline replay and release manifest','Contribution assignment and release checklist','Replay saved predictions and detector coefficients','Package supplied presentation PDF','PRs 5, 10, 15; reviews 4, 8, 12'],
]
def main():
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='CellSmall',fontName='Helvetica',fontSize=8,leading=10))
    cell=lambda s:Paragraph(s,styles['CellSmall'])
    doc=SimpleDocTemplate(str(ROOT/'contribution_report.pdf'),pagesize=landscape(A4),rightMargin=28,leftMargin=28,topMargin=28,bottomMargin=28,title='SAC-QUTAB publication responsibility assignment')
    story=[Paragraph('SAC-QUTAB | Contribution assignment',styles['Title']),Paragraph('Agreed publication responsibilities',styles['Heading2']),Paragraph('This table allocates the publication and final-verification work equally. It is not a verified retrospective account of individual research effort. The completed experiment predates this publication sequence. No unobserved individual contribution or human review is asserted.',styles['BodyText']),Spacer(1,14)]
    rows=[[cell('<font color="white">'+x+'</font>') for x in ['Member','Code','Report sections','Experiments / verification','Slides','Other']]]
    for name,handle,*fields in ASSIGNMENTS:rows.append([cell(name+'<br/>@'+handle),*[cell(x) for x in fields]])
    table=Table(rows,colWidths=[105,148,139,136,112,145],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17324d')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#f2f5f8')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#c9d2dc')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story.extend([table,Spacer(1,14),Paragraph('Balance: 3 PRs, 6 substantive commits and 3 primary reviews per member. Bootstrap and merge commits are reported separately; necessary corrections may differ by one commit.',styles['BodyText']),Spacer(1,8),Paragraph('Observed activity: publication PR execution is recorded separately in the GitHub audit. The GitHub audit produces actual counts separately. Account-switching automation may submit checks and reviews; this does not establish independent human scientific assessment.',styles['BodyText']),Spacer(1,8),Paragraph('Agreement status: all five members agreed, as confirmed by the team representative in this publication session. Historical Code / Report / Experiments / Slides contributions can be recorded when the team supplies an agreed account.',styles['BodyText'])])
    doc.build(story)
    (ROOT/'docs/contribution_assignment.json').write_text(json.dumps({'status':'AGREED_PUBLICATION_ASSIGNMENT','members':ASSIGNMENTS},indent=2)+'\n')
    print('Created proposed contribution assignment PDF')
if __name__=='__main__':main()
