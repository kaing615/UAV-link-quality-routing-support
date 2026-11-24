// Native PowerPoint reconstruction. Every mark, label, diagram node and rule is editable.
export function build(n,h,data){
 const {C,S,t,rect,line,arrow,dot,label,network,chain,drone,circle,shape}=h;
 const b={bold:true};
 const muted={color:C.gray};
 const mini=(x,y,w=1.7,h=.7,color=C.blue)=>network(x,y,w,h,{pts:[[0,.5],[.35,0],[.45,1],[1,.4]],edges:[[0,1],[0,2],[1,2],[1,3],[2,3]],r:.085,color,lineColor:C.gray});
 const doc=(x,y,w,hh,title)=>{rect(x+.1,y+.1,w,hh,C.bg,C.line);rect(x,y,w,hh,C.bg,C.gray);t(title,x+.2,y+.28,w-.4,.95,22,b);for(let i=0;i<3;i++)line(x+.2,y+hh-.55+i*.18,x+w-.25-i*.13,y+hh-.55+i*.18,C.line,2);};
 switch(n){
 case 19:{
   t('PDR:',.7,2.5,1.04,.48,25,{bold:true,color:C.blue});
   t('bao nhiêu gói đến đích?',1.75,2.5,5.75,.48,24);
   t('Chia cắt:',.7,3.92,1.75,.48,25,{bold:true,color:C.blue});
   t('mất đường nối bao lâu?',2.45,3.92,5,.48,24);
   t('Quãng đường:',.7,5.32,2.7,.48,25,{bold:true,color:C.blue});
   t('relay đi bao xa?',3.5,5.32,3.85,.48,24);
   line(.7,3.48,12.65,3.48); line(.7,4.88,12.65,4.88);
   for(let i=0;i<3;i++)rect(8.1+i*.47,2.59,.25,.25,C.blue);
   arrow(9.65,2.72,10.7,2.72,C.ink,1.7);
   rect(10.94,2.3,1.28,.8,C.bg,C.ink);rect(11.46,2.56,.25,.25,C.blue);line(10.8,3.18,12.36,3.18,C.ink,2.5);
   mini(8,3.83,1.05,.65,C.gray);mini(11.43,3.83,1.05,.65,C.gray);
   line(9.19,4.25,11.27,4.25,C.ink,1.5,true);
   t('×',9.94,4.02,.5,.53,34,{bold:true,color:C.blue,align:'center'});
   circle(10.2,3.77,.29,C.bg,C.blue);line(10.2,3.77,10.2,3.55,C.blue,1.6);line(10.2,3.77,10.35,3.86,C.blue,1.6);
   const p=chain(8.12,5.5,4.05,4,{lineColor:C.ink,r:.19});
   for(let i=0;i<3;i++)arrow(p[i][0]+.22,5.5,p[i+1][0]-.23,5.5,C.ink,1.5);
   line(8.12,6.07,12.17,6.07,C.blue,1.4);for(let i=0;i<=12;i++)line(8.12+i*4.05/12,6.07,8.12+i*4.05/12,5.95,C.blue,1.2);
   label('Mét di chuyển chưa phải năng lượng',.7,6.35,7.2,.45,21);
   break;
 }
 case 20:{
   t('+0,66',.75,2.55,5.5,1.45,88,{bold:true,color:C.blue});
   t('điểm phần trăm',.8,4.02,5.45,.7,36,{bold:true,color:C.blue});
   t('GNN: 96,01%',.8,5.08,5.3,.48,25,b);
   t('Chất lượng hiện tại: 95,35%',.8,5.67,5.55,.54,24,b);
   line(6.55,2.6,6.55,6.18);
   t('CI 95%: [0,38; 1,00]',7.1,2.97,5.45,.65,29,b);
   // Axis is linear in percentage points; estimate and CI endpoints share this scale.
   const axisX=7.6,scale=3.85,yy=4.82;
   arrow(7.15,5.58,12.5,5.58,C.gray,1);
   line(axisX,4.03,axisX,5.73,C.gray,1.1,true);
   t('0',axisX-.2,5.82,.4,.35,18,{align:'center',color:C.gray});
   const lo=axisX+.38*scale,hi=axisX+1*scale,est=axisX+.66*scale;
   line(lo,yy,hi,yy,C.blue,3);line(lo,yy-.16,lo,yy+.16,C.blue,2);line(hi,yy-.16,hi,yy+.16,C.blue,2);dot(est,yy,.115,C.blue);
   t('0,38',lo-.35,5.06,.7,.32,15,{align:'center',color:C.gray});
   t('1,00',hi-.35,5.06,.7,.32,15,{align:'center',color:C.gray});
   t('0,66',est-.35,4.31,.7,.35,18,{bold:true,align:'center',color:C.blue});
   t('40 kịch bản gradual, v8',.75,6.48,5,.3,14,muted);
   break;
 }
 case 21:{
   t('Giảm 0,20',.7,2.55,6.1,1.1,61,{bold:true,color:C.blue});
   t('giây chia cắt',.7,3.65,6.1,1.1,55,{bold:true,color:C.blue});
   t('Phản ứng trừ GNN',.75,5.08,5.6,.65,32,b);
   line(6.65,2.48,6.65,6.13,C.blue,2);
   t('CI 95%: [−1,05; 1,45]',7.1,2.64,5.45,.7,29,b);
   t('Khoảng chứa 0',7.1,3.44,5.4,.67,32,b);
   t('GNN: 31,25 giây',7.1,4.78,5.4,.6,29,b);
   t('Phản ứng: 31,45 giây',7.1,5.45,5.4,.6,29,b);
   t('40 kịch bản gradual, v9',.75,6.48,5,.3,14,muted);
   break;
 }
 case 22:{
   t('Đứng yên:',.75,2.27,1.95,.5,24,b);t('123,43 giây chia cắt',2.75,2.27,4.8,.5,24);
   const pts=[[0,.5],[.35,0],[.42,1],[.82,.4]];
   network(1.15,3.05,2.3,.95,{pts,edges:[[0,1],[0,2],[1,2],[1,3],[2,3]],r:.11});
   network(5.4,3.05,1.5,.95,{pts:[[0,.5],[.6,0],[1,1]],edges:[[0,1],[0,2],[1,2]],r:.11});
   line(.75,4.35,7.85,4.35);
   t('Phản ứng:',.75,4.48,1.95,.5,24,b);t('31,45 giây',2.75,4.48,4.8,.5,24);
   network(1.15,5.23,2.3,.95,{pts,edges:[[0,1],[0,2],[1,2],[1,3],[2,3]],r:.11});
   network(5.4,5.23,1.5,.95,{pts:[[0,.5],[.6,0],[1,1]],edges:[[0,1],[0,2],[1,2]],r:.11});
   // Join existing endpoint nodes; draw only the new central relay.
   line(3.036,5.61,4.22,5.66,C.line,1.5);line(4.22,5.66,5.4,5.705,C.line,1.5);dot(4.22,5.66,.11,C.blue);
   const xx=[1.6,2.6,3.4,4.2,5.15,6.5],ys=[6.46,6.46,6.24,6.24,6.46,6.46];
   for(let i=0;i<xx.length-1;i++)line(xx[i],ys[i],xx[i+1],ys[i+1],C.blue,1.3,true,i===xx.length-2);dot(xx[0],ys[0],.09,C.bg,C.blue);
   line(8.2,2.45,8.2,6.5);
   t('Tổng quãng đường:',8.65,2.94,3.95,.7,26,b);
   label('637,48 m',8.65,3.78,3.95,1.18,43);
   line(8.65,5.35,12.6,5.35);
   t('So sánh mô tả của v9',8.65,5.65,3.95,.75,25,muted);
   break;
 }
 case 23:{
   rect(.75,2.91,5.15,3.32,C.bg,C.blue,{line:{color:C.blue,width:1.8,dashType:'dash'}});
   network(1.2,3.43,2.3,1.35,{pts:[[0,.5],[.4,0],[.4,1],[.88,.3],[1,.95]],edges:[[0,1],[0,2],[1,2],[1,3],[2,3],[2,4],[3,4]],r:.12});
   arrow(3.7,4.08,4.07,4.08,C.blue,2);arrow(4.07,4.08,3.7,4.08,C.blue,2);
   rect(4.2,3.55,1.28,1.1,C.bg,C.blue);line(4.84,4.65,4.84,4.92,C.blue,2);line(4.42,4.95,5.26,4.95,C.blue,2);
   dot(4.45,4.28,.07);dot(5.19,3.84,.07);arrow(4.5,4.25,5.12,3.9,C.blue,1.2,true);
   label('Đã điều khiển tuyến và vị trí trong ns-3',.92,5.48,4.8,.49,16);
   drone(6.8,3.65,.78,C.gray);arrow(5.95,3.65,6.25,3.65,C.gray,1.2,true);
   rect(6.22,4.85,1.14,.65,C.bg,C.gray);circle(6.49,5.14,.1,C.bg,C.gray);circle(7.08,5.14,.1,C.bg,C.gray);line(6.8,4.85,6.8,4.56,C.gray,1);arrow(5.95,5.17,6.15,5.17,C.gray,1.2,true);
   line(7.72,2.75,7.72,6.5);
   const rows=[['Chưa bay thử UAV thật',2.82],['Chưa tích hợp autopilot',3.63],['Chưa dùng\nReinforcement Learning',4.43],['Một seed huấn luyện\nmô hình mới',5.51]];
   for(const [s,y]of rows){rect(8.08,y+.14,.23,.23,C.bg,C.gray);t(s,8.55,y,4.02,.78,23,b);}
   break;
 }
 case 24:{
   const ys=[2.44,3.52,4.6,5.68];
   const labels=['Dự báo tốt hơn chưa chắc mạng tốt hơn','Mạng tốt hơn chưa chứng minh bay thật','Di chuyển có ích chưa chứng minh GNN có ích','CI chứa 0: chưa đủ bằng chứng'];
   for(let i=0;i<4;i++){t(labels[i],5.65,ys[i],6.95,.76,i===2?21:23,b);if(i<3)line(.75,ys[i]+.95,12.6,ys[i]+.95);}
   const spark=(x,y,color,dashed)=>{line(x,y+.63,x+1.47,y+.63,C.gray,1);line(x,y+.63,x,y,C.gray,1);const p=[[x+.13,y+.5],[x+.48,y+.28],[x+.8,y+.48],[x+1.13,y+.2],[x+1.43,y+.04]];for(let i=0;i<4;i++)line(...p[i],...p[i+1],color,1.5,dashed);p.forEach(([a,b])=>dot(a,b,.045,dashed?C.bg:color,color));};
   spark(1.03,2.41,C.gray,true);arrow(2.75,2.77,3.16,2.77,C.blue,1.6);spark(3.46,2.41,C.blue,false);
   mini(1.02,3.65,1.45,.57,C.gray);arrow(2.75,3.94,3.16,3.94,C.blue,1.6);mini(3.47,3.65,1.45,.57);
   const path=(x,y,c,dash)=>{const ps=[[x,y+.42],[x+.5,y+.11],[x+.92,y+.47],[x+1.48,y+.18]];for(let i=0;i<3;i++)line(...ps[i],...ps[i+1],c,1.6,dash);dot(...ps[0],.075);dot(...ps[3],.075);};
   path(1.02,4.68,C.gray,true);arrow(2.75,5.02,3.16,5.02,C.blue,1.6);path(3.47,4.68,C.blue,false);
   line(1.6,6.12,4.52,6.12,C.blue,1.7);line(1.6,6.01,1.6,6.23,C.blue,1.2);line(4.52,6.01,4.52,6.23,C.blue,1.2);dot(2.88,6.12,.08);line(3.12,5.75,3.12,6.5,C.gray,1,true);t('0',2.98,6.49,.28,.23,12,{align:'center',color:C.gray});
   break;
 }
 case 25:{
   t('Mô hình dự báo',.85,2.15,3.7,.45,23,b);
   const p=[[1.07,3.48],[2.17,2.95],[2.17,3.99],[3.38,3.48]];
   for(const [a,bb]of [[0,1],[0,2],[1,3],[2,3]])arrow(...p[a],...p[bb],C.blue,1.5);
   p.forEach(([x,y])=>circle(x,y,.22,C.bg,C.gray));
   t('Kết quả và dữ liệu đối chiếu',.85,4.72,4.3,.43,21,b);
   line(1,6.35,1,5.34,C.gray,1);arrow(1,6.35,4.04,6.35,C.gray,1);
   const px=[1,1.45,1.87,2.31,2.75,3.18,3.62,3.92],py=[6.04,5.63,5.85,6.1,5.73,5.55,5.81,5.46];
   for(let i=0;i<px.length-1;i++){line(px[i],py[i],px[i+1],py[i+1],C.blue,1.6);line(px[i],py[i]+.12,px[i+1],py[i+1]+.12,C.gray,1,true);}
   rect(5.05,3.06,2.58,3.24,C.bg,C.gray);rect(4.9,2.91,2.58,3.24,C.bg,C.gray);
   t('Hồ sơ\nkiểm chứng',5.08,3.44,2.22,1.3,25,b);for(let i=0;i<3;i++)line(5.23,5.05+i*.23,7.13-i*.17,5.05+i*.23,C.line,2);
   t('Thí nghiệm vòng kín',9.02,2.15,3.65,.45,23,b);
   rect(9.05,2.87,1.3,.55,C.bg,C.gray);rect(11.22,2.87,1.3,.55,C.bg,C.gray);rect(10.15,3.93,1.3,.55,C.bg,C.gray);
   arrow(10.36,3.14,11.13,3.14,C.blue,1.6);line(12.52,3.14,12.72,3.14,C.blue,1.6);line(12.72,3.14,12.72,4.2,C.blue,1.6);arrow(12.72,4.2,11.48,4.2,C.blue,1.6);line(10.13,4.2,8.83,4.2,C.blue,1.6);line(8.83,4.2,8.83,3.14,C.blue,1.6);arrow(8.83,3.14,9.02,3.14,C.blue,1.6);
   doc(9.05,4.8,1.65,1.88,'Báo cáo\n58 trang');doc(10.88,4.8,1.65,1.88,'Bài báo\n12 trang');
   line(3.82,3.47,4.31,3.47,C.blue,1.6);line(4.31,3.47,4.31,3.97,C.blue,1.6);arrow(4.31,3.97,4.82,3.97,C.blue,1.6);
   line(4.05,5.6,4.31,5.6,C.blue,1.6);line(4.31,5.6,4.31,5.13,C.blue,1.6);arrow(4.31,5.13,4.82,5.13,C.blue,1.6);
   arrow(8.73,3.82,7.65,3.82,C.blue,1.6);arrow(8.86,5.65,7.65,5.65,C.blue,1.6);
   break;
 }
 case 26:{
   network(.9,3.13,2.17,2.15,{pts:[[0,.34],[.45,0],[.35,.5],[.02,.85],[.65,1],[1,.46]],edges:[[0,1],[0,2],[0,3],[1,2],[1,5],[2,3],[2,4],[2,5],[3,4],[4,5]],r:.13});
   line(3.34,4.17,3.75,4.17,C.gray,1.8);line(3.75,3.12,3.75,5.38,C.gray,1.8);arrow(3.75,3.12,4.18,3.12,C.gray,1.8);arrow(3.75,5.38,4.18,5.38,C.gray,1.8);
   // Route icon and motion marker are editable primitives.
   line(4.42,3.31,4.42,2.99,C.blue,2.5);line(4.42,2.99,4.98,2.99,C.blue,2.5);line(4.98,2.99,4.98,2.69,C.blue,2.5);circle(4.42,3.38,.095,C.bg,C.blue);circle(4.98,2.61,.095,C.bg,C.blue);
   t('Định tuyến:\nlợi ích từ làm mượt',5.35,2.64,3.32,1.25,22,b);
   circle(4.7,5.15,.28,C.blue);circle(4.7,5.15,.11,C.bg);shape(S.triangle,{x:4.46,y:5.28,w:.48,h:.49,rotate:180,fill:{color:C.blue},line:{color:C.blue}});
   t('Vị trí:\ngần giới hạn hình học',5.35,4.8,3.32,1.25,22,b);
   line(8.83,3.12,9.08,3.12,C.gray,1.8);line(9.08,3.12,9.08,5.38,C.gray,1.8);line(8.83,5.38,9.08,5.38,C.gray,1.8);arrow(9.08,4.17,9.44,4.17,C.gray,1.8);
   rect(9.65,3.43,2.96,1.55,C.blue);t('Bước tiếp:\nđộ bền theo\nseed (v11)',9.83,3.58,2.6,1.25,22,{bold:true,color:C.white});
   t('RL chỉ khi có\nđánh đổi thật',9.68,5.23,2.94,.84,21);
   break;
 }
 default: throw new Error(`Unexpected slide ${n}`);
 }
}
