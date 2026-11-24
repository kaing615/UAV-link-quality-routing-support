// Native PowerPoint reconstruction: no baked slide images or image overlays.
export function build(n,h,data) {
 const {C,S,t,rect,line,arrow,dot,label,network,chain,drone,circle,shape}=h;
 const center={align:'center',bold:true};
 switch(n) {
 case 1: {
  t('Dự báo liên kết\ncho mạng UAV',.72,2.0,6.35,1.85,44,{bold:true,breakLine:false});
  t('Định tuyến và bố trí UAV',.75,4.0,6.1,.56,25,{color:C.gray});
  line(.75,5.02,6.4,5.02,C.line,1);
  t('Nguyễn Đình Tâm · Võ Công Vinh',.75,5.35,6.2,.5,21,{bold:true});
  t('GVHD: ThS. Đặng Lê Bảo Chương',.75,5.87,6.2,.45,19,{color:C.gray});
  const p=[[9.7,2.15],[7.95,3.85],[11.65,3.85],[8.5,5.55],[11.1,5.55]];
  for(const [a,b] of [[0,1],[0,2],[1,2],[1,3],[2,4],[3,4]])line(...p[a],...p[b],C.blue,1.5,true);
  p.forEach(([x,y])=>{drone(x,y,.95,C.ink);for(let k=0;k<3;k++){const r=.22+k*.12;const arc=Array.from({length:9},(_,j)=>{const a=(-135+j*90/8)*Math.PI/180;return[x+r*Math.cos(a),y-.22+r*Math.sin(a)];});for(let j=0;j<8;j++)line(...arc[j],...arc[j+1],C.blue,1.8);}});
  break;
 }
 case 2: {
  const xs=[1.55,4.9,8.25,11.6];
  for(let i=0;i<3;i++)arrow(xs[i]+.68,4.05,xs[i+1]-.72,4.05,C.line,2);
  ['Hiểu bài toán','Học dự báo','Thử điều khiển','Đo kết quả'].forEach((v,i)=>{circle(xs[i],4.05,.68,C.blue);t(`0${i+1}`,xs[i]-.52,3.63,1.04,.84,36,{...center,color:C.white});const width=i===2?2.8:2.2;t(v,xs[i]-width/2,5.1,width,.75,23,center);});
  break;
 }
 case 3: {
  const xs=[1.5,4,6.5,9,11.5], yy=4.25;
  for(let i=0;i<4;i++)arrow(xs[i]+.7,yy,xs[i+1]-.7,yy,C.blue,1.8,i===2);
  xs.forEach(x=>{circle(x,yy,.67,C.pale);drone(x,yy,.75);});
  t('Nguồn gửi',.65,5.85,1.75,.5,22,center);t('Đích nhận',10.6,5.85,1.85,.5,22,center);
  line(3.55,5.68,3.55,5.86);line(3.55,5.86,9.45,5.86);line(9.45,5.68,9.45,5.86);t('UAV trung chuyển',4.3,6.02,4.4,.5,22,center);
  t('Liên kết yếu',7.55,2.6,3.4,.5,23,{bold:true,color:C.blue});line(9.2,3.25,8.0,3.25,C.blue,1.5);arrow(8,3.25,7.65,3.95,C.blue,1.5);
  label('Gói tin bị mất',6.75,5.13,2.65,.5,20);
  break;
 }
 case 4: {
  t('Định tuyến: đổi đường gói tin',.72,2.33,5.65,.55,23,{bold:true});
  t('Vị trí: di chuyển relay',7.0,2.33,5.55,.55,23,{bold:true});line(6.65,2.5,6.65,6.1,C.line,1);
  network(1.1,3.25,4.85,2.15,{path:[[0,1],[1,4],[4,5]],color:C.line,r:.17});
  network(7.65,3.25,4.55,2.15,{hollow:[3],color:C.line,r:.17});dot(7.65,4.325,.18);dot(12.2,4.325,.18);
  // Move the center relay to the right; origin and target remain separate selectable objects.
  arrow(10.46,4.22,10.73,4.22,C.blue,2.2);dot(10.95,4.22,.17);
  line(.72,6.22,12.55,6.22,C.line,1);
  t('Hai tác động được kiểm tra riêng',.85,6.37,11.8,.45,23,{bold:true});
  break;
 }
 case 5: {
  const xs=[1.5,4.05,6.55,9.05,11.6],yy=4;
  for(let i=0;i<4;i++)arrow(xs[i]+.85,yy,xs[i+1]-.9,yy,C.ink,1.4);
  network(.9,3.2,1.2,1.6,{pts:[[0,.8],[.2,.15],[.6,0],[1,.45],[.55,1],[.42,.52]],edges:[[0,1],[1,2],[2,3],[3,4],[4,0],[0,5],[1,5],[2,5],[3,5],[4,5]],r:.1,lineColor:C.ink});
  for(let j=0;j<3;j++){rect(3.3,3.25+j*.55,1.5,.4,C.pale);dot(3.48,3.45+j*.55,.08);const base=3.45+j*.55;const ys=[0,-.07,.04,-.05,.07,0];for(let k=0;k<5;k++)line(3.73+k*.17,base+ys[k],3.73+(k+1)*.17,base+ys[k+1],C.ink,1);}
  arrow(5.9,4.75,7.28,4.75,C.ink,1.4);arrow(5.9,4.75,5.9,3.18,C.ink,1.4);const p=[[5.94,4.68],[6.1,4.3],[6.35,4.0],[6.65,3.76],[6.95,3.62],[7.28,3.58]];for(let k=0;k<p.length-1;k++)line(...p[k],...p[k+1],C.blue,2.1);
  dot(8.35,4,.11);for(const y of [3.3,4,4.7]){arrow(8.5,4,9.65,y,y===4?C.blue:C.ink,1.4);dot(9.8,y,.11,y===4?C.blue:C.bg,y===4?C.blue:C.ink);}
  line(10.85,4.73,12.4,4.73,C.ink,1.4);[.45,.72,.95].forEach((v,i)=>rect(11+i*.45,4.65-v,.3,v,i<2?C.blue:C.bg,C.blue));dot(11.6,3.15,.1);line(11.6,3.25,11.6,3.7,C.ink,1.3);
  data.labels.forEach((v,i)=>t(v,xs[i]-1.15,5.22,2.3,.68,18,center));
  break;
 }
 case 6: {
  t('210',.72,2.45,4.6,1.8,106,{bold:true,color:C.blue});t('lượt mô phỏng',.8,4.22,4.4,.6,31,{bold:true});line(.8,5.05,5.0,5.05,C.line,1);t('20 UAV mỗi lượt',.8,5.26,4.5,.55,25,{bold:true});t('Tốc độ · mật độ · tải mạng',.8,5.95,4.6,.5,20);
  line(5.4,2.45,5.4,6.45,C.line,1);t('21 điều kiện × 10 seed',5.88,2.35,6.6,.6,29,{bold:true});
  const pts=Array.from({length:20},(_,i)=>{const a=i*Math.PI*2/20;const r=i%3===0?1.0:1.45;return[7.7+Math.cos(a)*r,4.7+Math.sin(a)*r];});
  for(let i=0;i<20;i++){line(...pts[i],...pts[(i+1)%20],C.line,.8);line(...pts[i],...pts[(i+6)%20],C.line,.6);}pts.forEach(p=>dot(...p,.08));
  for(let r=0;r<21;r++)for(let c=0;c<10;c++)rect(10.27+c*.21,3.28+r*.145,.15,.1,C.line);
  break;
 }
 case 7: {
  network(.9,3.2,2.15,2.25,{r:.19,color:C.line});dot(1.5,3.38,.2);dot(2.13,4.21,.2);dot(2.62,5.18,.2);
  arrow(3.45,4.35,3.9,4.35,C.blue,2.3);
  network(4.25,3.2,2.15,2.25,{r:.19,color:C.line});dot(4.85,3.38,.2);dot(5.48,4.21,.2);dot(5.97,5.18,.2);dot(6.4,4.33,.2);
  line(6.86,2.75,6.86,5.95,C.line,1);t('QoS tại mốc 5 bước',7.18,2.96,5.15,.6,27,{bold:true,color:C.gray});label('GNN: Macro-F1 0,845',7.18,3.82,5.18,.75,26);label('XGBoost: 0,784',7.18,4.86,5.18,.75,26,C.pale);
  line(.72,6.2,12.55,6.2,C.line,1);t('Cảnh báo nhầm ổn định vẫn còn',.85,6.35,11.5,.5,24,{bold:true});
  break;
 }
 case 8: {
  network(1.05,3.15,2.4,1.8,{pts:[[0,.7],[.35,0],[.9,.3],[.65,1]],edges:[[0,1],[1,2],[2,3],[3,0]],r:.12,color:C.line});arrow(1.08,4.4,3.18,3.75,C.blue,1.8,true);t('×',3.02,3.5,.5,.5,30,{color:C.blue,bold:true});
  arrow(3.9,4.15,4.45,4.15,C.blue,2);
  network(5.18,3.2,2.0,1.72,{pts:[[0,.65],[.5,0],[1,.3],[.55,1]],edges:[[0,1],[1,2],[2,3],[3,0]],r:.1,color:C.line});arrow(5.25,4.25,7.08,3.75,C.blue,1.8,true);shape(S.ellipse,{x:4.92,y:2.87,w:2.7,h:2.55,fill:{color:C.bg,transparency:100},line:{color:C.blue,width:4}});line(7.1,5.0,7.75,5.64,C.blue,9);
  arrow(8.0,4.15,8.55,4.15,C.blue,2);
  network(9.18,3.15,2.65,1.8,{pts:[[0,.7],[.35,0],[.75,.18],[1,.7],[.65,1],[.3,.95],[.35,.45],[.68,.45]],edges:[[0,1],[1,2],[2,3],[0,5],[5,4],[4,3],[0,6],[6,7],[7,3]],path:[[0,6],[6,7],[7,3]],r:.11,color:C.line});
  t('v2–v3: chưa đạt mục tiêu',.65,5.65,3.4,.65,18,center);t('Kiểm tra đường đi thực tế',4.62,5.65,3.55,.65,18,center);t('v4–v7: kiểm chứng định tuyến',8.76,5.65,3.85,.65,18,center);label('Rà soát cả môi trường mô phỏng',3.2,6.42,7.0,.44,20);
  break;
 }
 case 9: {
  rect(5.35,2.18,2.65,1.02,C.blue);const p=[[5.75,2.87],[6.1,2.48],[6.5,2.7],[7.3,2.43]];for(let i=0;i<3;i++)line(...p[i],...p[i+1],C.white,2);p.forEach(([x,y])=>dot(x,y,.07,C.blue,C.white));
  line(5.35,2.69,3.2,2.69,C.line,1.5);arrow(3.2,2.69,3.2,3.5,C.gray,1.5);line(8,2.69,10.1,2.69,C.line,1.5);arrow(10.1,2.69,10.1,3.5,C.gray,1.5);
  label('v8: dự báo cho định tuyến',.9,3.65,5.15,.58,23);label('v9: dự báo cho di chuyển',7.13,3.65,5.15,.58,23);
  network(1.25,4.55,4.7,1.3,{path:[[0,3],[3,4],[4,5]],color:C.ink,r:.12});t('Relay đứng yên',1.3,6.14,4.65,.55,24,center);
  line(7.7,5.45,9.45,4.88,C.gray,1.5,true);line(9.45,4.88,11.7,5.45,C.gray,1.5,true);dot(7.7,5.45,.18);dot(11.7,5.45,.18);dot(9.45,4.88,.18,C.bg,C.blue);t('Giữ cùng luật định tuyến',7.12,6.14,5.3,.55,24,center);
  break;
 }
 default: throw new Error(`Unknown slide ${n}`);
 }
}
