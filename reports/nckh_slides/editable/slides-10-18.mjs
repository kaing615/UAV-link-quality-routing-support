// Native PowerPoint reconstruction; no raster visual content.
export function build(n,h,data) {
 const {C,S,t,rect,line,arrow,dot,label,network,chain,drone,circle,shape}=h;
 const txt=(s,x,y,w,hh=.5,sz=22,o={})=>t(s,x,y,w,hh,sz,{bold:true,...o});
 const graph=(x,y,w,hh,opts={})=>network(x,y,w,hh,opts);
 const gridDots=(x,y,cols,rows,color=C.gray,dx=.29,dy=.32,r=.08)=>{for(let j=0;j<rows;j++)for(let i=0;i<cols;i++)dot(x+i*dx,y+j*dy,r,color);};
 const radial=(x,y,r,halo=false)=>{if(halo)circle(x,y,r*1.2,C.lightblue);const p=[];for(let i=0;i<6;i++){const a=i*Math.PI/3;p.push([x+Math.cos(a)*r,y+Math.sin(a)*r]);}p.forEach(([a,b])=>line(x,y,a,b,C.gray,1.15));p.forEach(([a,b])=>dot(a,b,.11));dot(x,y,.13);};
 switch(n){
 case 10:{
   data.labels.forEach((s,i)=>{rect(.72,2.45+i*.72,.18,.18,C.blue);t(s,1.12,2.30+i*.72,3.5,.5,22);});
   line(4.9,2.2,4.9,6.3,C.line);
   for(const [count,y] of [[5,3.3],[7,5.3]]){
     const x=5.7,w=6.5,step=w/(count-1);
     for(let i=0;i<count-1;i++){
       arrow(x+i*step+.24,y,x+(i+1)*step-.26,y,C.ink,1.2);
       for(let k=0;k<3;k++)rect(x+i*step+.36+k*.14,y-.28,.09,.09,C.blue);
       // Dashed wireless hop, independent line segments.
       const px=x+i*step, qx=x+(i+1)*step;
       line(px+.1,y-.25,(px+qx)/2,y-.65,C.gray,1,true);
       line((px+qx)/2,y-.65,qx-.1,y-.25,C.gray,1,true);
     }
     for(let i=0;i<count;i++)dot(x+i*step,y,.23);
     txt('S',x-.16,y-.2,.32,.4,21,{color:C.white,align:'center'});
     txt('D',x+w-.16,y-.2,.32,.4,21,{color:C.white,align:'center'});
   }
   break;
 }
 case 11:{
   // Observations on the left, learned graph in the middle, forecast gauge at right.
   const ys=[2.8,3.9,5.0];
   circle(1.03,2.85,.17,C.blue);arrow(1.4,2.85,1.85,2.85,C.blue,2);
   dot(.85,3.93,.13);dot(1.6,3.93,.13);line(.98,3.93,1.47,3.93,C.blue,2);
   line(.85,5.12,1.21,4.78,C.gray);line(1.21,4.78,1.6,5.12,C.gray);dot(.85,5.12,.12,C.gray);dot(1.21,4.78,.13);dot(1.6,5.12,.12,C.gray);
   data.labels.slice(0,3).forEach((s,i)=>t(s,2.05,ys[i]-.18,2.75,.48,20,{bold:true}));
   line(4.82,2.88,5.08,2.88,C.gray,1.5);line(4.82,5.05,5.08,5.05,C.gray,1.5);line(5.08,2.88,5.08,5.05,C.gray,1.5);arrow(5.08,3.95,5.62,3.95,C.gray,1.6);
   graph(5.85,2.75,2.2,2.6,{pts:[[.5,0],[0,.4],[.15,1],[.85,1],[1,.4]],edges:[[0,1],[0,4],[1,2],[2,3],[3,4],[1,3],[2,4]],r:.19,lineColor:C.gray});
   arrow(8.35,4.05,8.95,4.05,C.gray,1.5);
   txt('Tỷ lệ nhận probe\ndự báo',9.3,2.4,3.15,.85,25,{align:'center'});
   const cx=10.88,cy=5.26,rr=1.42;
   for(let i=0;i<36;i++){const a=Math.PI+i*Math.PI/36,b=Math.PI+(i+1)*Math.PI/36;line(cx+rr*Math.cos(a),cy+rr*Math.sin(a),cx+rr*Math.cos(b),cy+rr*Math.sin(b),i<22?C.blue:C.line,11);}
   line(cx,cy,cx+.62,cy-.87,C.blue,4);dot(cx,cy,.16);
   line(.7,6.02,12.6,6.02,C.line);arrow(.75,6.4,1.2,6.4,C.blue,2.5);txt('Chỉ dùng hiện tại và quá khứ',1.5,6.13,9.9,.55,24);
   break;
 }
 case 12:{
   dot(.9,2.65,.13);t('Nút là UAV',1.18,2.44,2,.4,17);
   line(.77,3.16,1.03,3.16,C.gray,1.2);t('Cạnh là liên kết',1.18,2.97,2,.4,17);
   label('GNN trao đổi\nthông tin\nhàng xóm',.68,3.66,2.34,1.25,21);
   line(3.28,2.3,3.28,5.02,C.line);
   radial(4.43,3.58,.62,false);arrow(5.15,3.58,5.55,3.58,C.gray,1.4);
   radial(6.43,3.58,.62,true);label('Vòng 1',5.78,4.68,1.3,.45,19);
   arrow(7.18,3.58,7.57,3.58,C.gray,1.4);
   radial(8.47,3.58,.74,true);label('Vòng 2',7.82,4.68,1.3,.45,19);
   arrow(9.43,3.58,9.9,3.58,C.gray,1.4);dot(10.45,3.58,.16);dot(11.9,3.58,.16);line(10.64,3.58,11.71,3.58,C.blue,1.7,true);
   label('Dự đoán liên kết',10.05,4.68,2.23,.45,17);
   line(.65,5.3,12.65,5.3,C.line);
   label('MLP bỏ\ntrao đổi\nhọc được',.68,5.58,2.34,1.03,20);
   for(let j=0;j<2;j++)for(let i=0;i<6;i++)rect(3.82+i*.31,5.64+j*.5,.31,.34,[0,1,4].includes(i)?C.lightblue:C.pale,C.gray);
   arrow(5.87,6.08,6.55,6.08,C.gray,1.5);label('MLP',6.72,5.77,1.42,.62,21);
   arrow(8.36,6.08,9.13,6.08,C.gray,1.5);dot(9.7,6.08,.15);dot(11.15,6.08,.15);line(9.88,6.08,10.97,6.08,C.blue,1.7,true);
   break;
 }
 case 13:{
   rect(.72,2.75,4.6,3.03,C.bg,C.line,{line:{color:C.line,width:1,dashType:'dash'}});
   txt('48 lượt cũ để học',1.02,3.02,4,.5,25);gridDots(1.12,3.8,12,2,C.gray,.33,.33,.092);
   txt('20 lượt cũ chọn mô hình',1.02,4.7,4,.5,23);gridDots(1.12,5.43,12,1,C.gray,.33,.33,.092);
   arrow(5.5,4.28,6.05,4.28,C.blue,2.2);
   // Lock is assembled from editable geometry.
   shape(S.ellipse,{x:6.47,y:3.26,w:.8,h:1.13,fill:{color:C.bg,transparency:100},line:{color:C.blue,width:5}});
   rect(6.25,3.9,1.24,1.11,C.blue);dot(6.87,4.32,.1,C.white);rect(6.82,4.32,.1,.32,C.white);
   txt('Khóa trọng số',5.47,5.2,2.8,.48,22,{align:'center'});
   arrow(7.74,4.28,8.3,4.28,C.blue,2.2);
   rect(8.57,3.28,3.98,2.04,C.bg,C.line,{line:{color:C.line,width:1,dashType:'dash'}});
   txt('v8–v9 dùng seed mới',8.84,3.58,3.5,.55,24);
   gridDots(9.02,4.43,8,2,C.blue,.43,.4,.11);
   break;
 }
 case 14:{
   // Native segmented elliptical arrows form the closed feedback loop.
   const cx=6.65,cy=4.48,rx=3.15,ry=1.75;
   const angles=[-Math.PI/2,-.3,Math.PI*.29,Math.PI*.73,Math.PI*1.1];
   const pos=angles.map(a=>[cx+rx*Math.cos(a),cy+ry*Math.sin(a)]);
   for(let i=0;i<5;i++){
     let a=angles[i]+.075,b=(i===4?angles[0]+2*Math.PI:angles[i+1])-.075;
     for(let k=0;k<18;k++){const p=a+(b-a)*k/18,q=a+(b-a)*(k+1)/18;line(cx+rx*Math.cos(p),cy+ry*Math.sin(p),cx+rx*Math.cos(q),cy+ry*Math.sin(q),C.blue,2,false,k===17);}
   }
   pos.forEach(([x,y])=>dot(x,y,.14));
   txt('Quan sát mỗi giây',4.95,2.12,3.4,.5,24,{align:'center'});
   txt('Dự báo',10.03,3.54,2,.55,24);
   txt('Quyết định',9.52,5.51,2.7,.55,24);
   txt('Thực thi trong\nmô phỏng',1.02,5.48,3.2,.85,23,{align:'right'});
   txt('Đo lại\ntrạng thái',1.17,3.3,2.2,.83,23,{align:'right'});
   const p=[[6.65,3.55],[5.25,4.48],[8.05,4.48],[6.65,5.42]];
   for(const [a,b] of [[0,1],[0,2],[1,2],[1,3],[2,3]])line(...p[a],...p[b],C.gray,1.2,true);
   p.forEach(([x,y])=>drone(x,y,.8));
   break;
 }
 case 15:{
   const yy=[2.47,3.38,4.29];
   data.labels.slice(0,3).forEach((s,i)=>{rect(.75,yy[i],3.65,.65,C.pale);txt(s,.97,yy[i]+.07,3.2,.5,22);line(4.4,yy[i]+.325,5.04,yy[i]+.325,C.gray,1.5);});
   line(5.04,2.795,5.04,4.615,C.gray,1.5);arrow(5.04,3.705,5.65,3.705,C.gray,1.5);
   rect(5.78,2.66,2.7,2.14,C.blue);
   txt('Relay giữ\nnguyên vị trí',6.0,2.85,2.26,.9,24,{color:C.white,align:'center'});
   const p=[[6.48,4.23],[7.13,3.93],[7.8,4.23],[7.13,4.57]];for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[1,3]])line(...p[a],...p[b],C.white,1.6);p.forEach(([x,y])=>dot(x,y,.09,C.blue,C.white));
   arrow(8.64,3.7,9.12,3.7,C.gray,1.5);graph(9.42,2.65,2.75,2.1,{pts:[[0,.5],[.42,0],[.42,.58],[.32,1],[.95,.29],[.85,.89]],edges:[[0,1],[0,2],[0,3],[1,2],[1,4],[2,3],[2,4],[3,5],[4,5]],r:.16,lineColor:C.gray});
   line(.65,5.12,12.65,5.12,C.line);
   rect(.75,5.68,4.75,.66,C.pale);txt('Đối chứng OLSR-ETX nhanh',.95,5.76,4.35,.5,22);
   arrow(5.65,6.0,8.15,6.0,C.gray,1.5);graph(8.5,5.45,2.15,1.18,{dash:true,r:.11});
   break;
 }
 case 16:{
   txt('Đứng yên để đối chiếu',.72,2.4,4.2,.55,23);
   chain(5.15,2.67,6.6,5,{r:.13});
   txt('Kích hoạt bằng số đo hoặc dự báo',.72,3.6,6.0,.6,23);
   arrow(6.75,3.9,7.45,3.9,C.blue,4);
   txt('Cùng luật bố trí relay',7.8,3.6,4.8,.6,23);
   const p=chain(2.2,5.5,8.95,5,{r:.14});
   for(let i=1;i<=3;i++){const [x,y]=p[i];dot(x,y-.86,.13,C.bg,C.blue);arrow(x,y-.67,x,y-.2,C.ink,1.5);}
   txt('Tối đa 8 m/s',.72,6.08,4.05,.6,29);
   txt('Ngân sách: 400 m mỗi relay',5.25,6.08,7.3,.6,28);
   break;
 }
 case 17:{
   // Three functional layers, all built from native shapes.
   shape(S.parallelogram,{x:.8,y:2.3,w:4.2,h:1.15,fill:{color:C.blue},line:{color:C.blue}});
   const p=[[2.05,2.88],[2.75,2.52],[3.5,2.88],[3.03,3.22],[2.48,3.22]];
   for(let i=0;i<p.length;i++)line(...p[i],...p[(i+1)%p.length],C.white,1.6);p.forEach(([x,y])=>dot(x,y,.11,C.white));
   arrow(2.9,3.54,2.9,3.93,C.ink,1.7);
   shape(S.parallelogram,{x:.8,y:4.02,w:4.2,h:1.05,fill:{color:C.pale},line:{color:C.pale}});
   rect(2.2,4.28,1.35,.5,C.bg,C.ink);arrow(1.55,4.53,2.08,4.53,C.ink,1.7);arrow(3.68,4.53,4.17,4.53,C.ink,1.7);
   arrow(2.9,5.16,2.9,5.55,C.ink,1.7);
   shape(S.parallelogram,{x:.8,y:5.65,w:4.2,h:1.0,fill:{color:C.pale},line:{color:C.pale}});
   line(2.15,6.42,3.74,6.42,C.ink,1.5);line(2.15,6.42,2.15,5.86,C.ink,1.5);for(let i=0;i<3;i++)rect(2.4+i*.4,6.38-(.2+i*.13),.25,.2+i*.13,C.blue);
   const tools=['ns-3:','PyTorch:','Python, NumPy:','Ubuntu:','Matplotlib, LaTeX:'];
   const roles=['mô phỏng mạng','học GNN và MLP','điều khiển và phân tích','chạy thí nghiệm','hình và báo cáo'];
   for(let i=0;i<5;i++){const y=2.4+i*.83;txt(tools[i],5.6,y,3.0,.5,23,{color:C.blue});t(roles[i],8.8,y,3.8,.5,22);if(i<4)line(5.6,y+.68,12.65,y+.68,C.line);}
   break;
 }
 case 18:{
   txt('Mỗi nghiên cứu',.72,2.3,4.12,.4,20);
   line(.72,2.88,4.78,2.88,C.line);
   txt('80',.72,3.02,1.65,.85,53,{color:C.blue});t('lượt\nvalidation',2.62,3.07,2.1,.75,23,{bold:true});
   line(.72,4.05,4.78,4.05,C.line);
   txt('192',.72,4.17,1.92,.85,53,{color:C.blue});t('lượt test',2.83,4.27,1.85,.6,23,{bold:true});
   line(.72,5.19,4.78,5.19,C.line);
   txt('40 kịch bản gradual ghép cặp',.72,5.32,4.22,.48,19);
   txt('Khóa mô hình trước test',.72,5.88,4.22,.42,19);
   txt('Khoảng tin cậy 95%',.72,6.35,4.22,.38,19);
   line(5.06,2.3,5.06,6.55,C.line);
   const x0=5.35,y0=2.95,cw=1.42,rh=.8;
   for(let row=0;row<4;row++)for(let col=0;col<5;col++){
     const x=x0+col*cw,y=y0+row*rh;
     rect(x,y,cw,rh,row===0?(col===0?C.pale:C.blue):row===2?C.lightblue:C.bg,C.line);
     if(row===0&&col>0)t(`Phương\npháp ${String.fromCharCode(64+col)}`,x+.08,y+.08,cw-.16,rh-.16,17,{bold:true,color:C.white,align:'center'});
     if(row>0&&col===0)t(`Kịch bản ${row}`,x+.1,y+.1,cw-.2,rh-.2,16,{bold:true});
     if(row>0&&col>0)dot(x+cw/2,y+rh/2,.105);
   }
   break;
 }
 default:throw new Error(`Unsupported slide ${n}`);
 }
}
