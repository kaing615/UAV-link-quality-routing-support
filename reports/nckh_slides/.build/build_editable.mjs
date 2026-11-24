import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import pptxgen from 'pptxgenjs';
import JSZip from 'jszip';
import {helpers,C} from '../editable/helpers.mjs';
import {build as a} from '../editable/slides-01-09.mjs';
import {build as b} from '../editable/slides-10-18.mjs';
import {build as c} from '../editable/slides-19-26.mjs';
import {build as x} from '../editable/slides-extra.mjs';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const out=path.join(root,'editable');
const data=JSON.parse(await fs.readFile(path.join(root,'slides.json'),'utf8'));
const deck=new pptxgen();deck.layout='LAYOUT_WIDE';deck.author='Nguyễn Đình Tâm · Võ Công Vinh';deck.title='Dự báo liên kết cho mạng UAV — bản chỉnh sửa';deck.subject='Bản PowerPoint chỉnh sửa từ PDF 26 trang';deck.lang='vi-VN';deck.theme={headFontFace:'Arial',bodyFontFace:'Arial',lang:'vi-VN'};
const records=[];
// d.n identifies the original builder/source image; extra slides use d.key. Shown page number = position.
for(const [i,d] of data.entries()){
 const pos=i+1,id=d.key??String(d.n).padStart(2,'0');
 const s=deck.addSlide();s.name=`${String(pos).padStart(2,'0')} ${d.title}`.slice(0,100);s.background={color:C.bg};s.addNotes(d.note);
 const record={slide:pos,canvas:[1920,1080],background:'#'+C.bg,route:'native-components',texts:[],shapes:[]};const h=helpers(s,deck,record);
 const tw=Math.max(1.5,Math.min(5.1,.4+d.tag.length*.16));
 h.rect(.65,.48,tw,.42,C.blue);h.t(d.tag,.77,.5,tw-.24,.38,16,{color:C.white,bold:true});
 if(d.n!==1){const sz=d.title.length>63?28:d.title.length>53?30:34;h.t(d.title,.65,1.12,12.03,.86,sz,{bold:true,valign:'top'});}
 if(d.key||d.n<20||d.n===23||d.n===25||d.n===26)h.line(.65,7.04,12.35,7.04,C.line,.65);
 h.t(String(pos).padStart(2,'0'),12.45,7.05,.28,.22,10,{color:C.gray,align:'right'});
 (d.key?x:d.n<=9?a:d.n<=18?b:c)(d.key??d.n,h,d);
 records.push(record);
 const dir=path.join(out,'subagents',`slide-${id}`);await fs.mkdir(path.join(dir,'visual-layers'),{recursive:true});await fs.mkdir(path.join(dir,'qa'),{recursive:true});
 if(!d.key)await fs.copyFile(path.join(root,'slides',`slide-${id}.png`),path.join(dir,'source.png'));
 const texts=record.texts.map((t,i)=>({name:`text-${i+1}`,text:t.text,x:t.x*144,y:t.y*144,w:t.w*144,h:t.h*144,size:t.fontSize*2,typeface:t.fontFace,color:'#'+t.color,bold:!!t.bold,align:t.align||'left',valign:'middle',zIndex:10}));
 await fs.writeFile(path.join(dir,'text-layer.json'),JSON.stringify({...record,texts,shapes:undefined},null,2));
 await fs.writeFile(path.join(dir,'visual-layers','manifest.json'),JSON.stringify({canvas:[1920,1080],background:'#'+C.bg,route:'native-components',assets:[],nativeShapes:record.shapes},null,2));
 await fs.writeFile(path.join(dir,'qa','qa.md'),`# Slide ${d.n}\nNative PowerPoint reconstruction: all text, geometry, and diagram components are editable. No bitmap layers. Font Arial; icons simplified to native geometry. Reference source.png remains unchanged.\n`);
}
await fs.writeFile(path.join(out,'native-object-manifest.json'),JSON.stringify(records,null,2));
const raw=path.join(out,'editable.raw.pptx'),candidate=path.join(out,'editable.candidate.pptx');
await deck.writeFile({fileName:raw,compression:true});
const zip=await JSZip.loadAsync(await fs.readFile(raw));const ct=await zip.file('[Content_Types].xml').async('string');zip.file('[Content_Types].xml',ct.replace(/<Override\b[^>]*PartName="([^"]+)"[^>]*\/>/g,(entry,name)=>zip.file(name.replace(/^\//,''))?entry:''));
await fs.writeFile(candidate,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
const stats=[];for(let n=1;n<=data.length;n++){const x=await zip.file(`ppt/slides/slide${n}.xml`).async('string');stats.push({slide:n,textRuns:(x.match(/<a:t>/g)||[]).length,shapes:(x.match(/<p:sp>/g)||[]).length,pictures:(x.match(/<p:pic>/g)||[]).length,charts:(x.match(/<c:chart /g)||[]).length});}
if(stats.some(s=>s.textRuns<3||s.pictures>0))throw Error('Editability audit failed');
await fs.writeFile(path.join(out,'editability-audit.json'),JSON.stringify({slideCount:data.length,bitmapObjects:0,slides:stats},null,2));
console.log(JSON.stringify({candidate,stats}));
if(process.argv.includes('--finalize')){
 const skill='C:/Users/kaing/.codex/plugins/cache/openai-primary-runtime/presentations/26.1119.11507/skills/presentations';const runtime='C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies';process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
 const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
 const finalPath=path.join(root,'final','NCKH_UAV_bao_cao_editable.pptx');
 console.log(JSON.stringify(await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:runtime+'/python/python.exe',explicitTotalSlideCount:data.length,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Arial']},integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000'],verifyArtifactToolImport:true,receiptPath:path.join(out,'validation.json')})));
}
