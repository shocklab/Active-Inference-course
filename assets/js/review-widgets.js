/* Small finite examples accompanying the course's inference distinctions. */
(function () {
  'use strict';
  var A=window.AIF,el=A.el;
  function table(headers,rows) {
    var t=el('table',{style:'width:100%;font-variant-numeric:tabular-nums'});
    t.appendChild(el('thead',{},[el('tr',{},headers.map(function(h){return el('th',{text:h});}))]));
    t.appendChild(el('tbody',{},rows.map(function(row){return el('tr',{},row.map(function(v){return el('td',{text:String(v)});}));})));
    return el('div',{class:'tablewrap',style:'overflow-x:auto'},[t]);
  }
  A.register('chain-messages',function(mount){
    var n=A.slider({label:'Number of hidden states',min:3,max:20,step:1,value:3,decimals:0});
    var rel=A.slider({label:'P(report 1 | final state 1)',min:0.1,max:0.9,step:0.05,value:0.9});
    var out=A.readout(),body=el('div');
    function render(){
      var count=n.get(),p=rel.get(),m=[1-p,p],rows=[[count,'sensor likelihood',m[0].toFixed(6),m[1].toFixed(6)]];
      for(var i=count-1;i>=1;i--){
        m=[0.7*m[0]+0.3*m[1],0.2*m[0]+0.8*m[1]];
        if(i>=count-3||i<=3)rows.push([i,'backward message',m[0].toFixed(6),m[1].toFixed(6)]);
        else if(i===count-4)rows.push(['…','intermediate messages omitted','…','…']);
      }
      var ev=0.4*m[0]+0.6*m[1],brute=null;
      if(count<=12){
        brute=0;
        for(var bits=0;bits<Math.pow(2,count);bits++){
          var state=bits&1,w=state?0.6:0.4;
          for(var j=1;j<count;j++){var next=(bits>>j)&1;w*=state?(next?0.8:0.2):(next?0.3:0.7);state=next;}
          brute+=w*(state?p:1-p);
        }
      }
      body.replaceChildren(table(['Boundary index','Quantity','Given state 0','Given state 1'],rows));
      out.show([['Evidence',ev.toFixed(6)],['Assignments',Math.pow(2,count).toLocaleString()],['Enumeration difference',brute===null?'not enumerated above 12 states':Math.abs(brute-ev).toExponential(2)]]);
    }
    mount.appendChild(A.panel([A.row([n.el,rel.el]),out,body,A.note('Transition columns are (0.70, 0.30) and (0.20, 0.80); initial probabilities are (0.40, 0.60). Those initial probabilities are stationary for the transitions, so every state has them and the evidence does not change with the number of states. The other sensor likelihood is one minus its slider value. At reliability 0.50, the report is uninformative.')]));
    n.onchange(render);rel.onchange(render);render();
  });
  A.register('belief-evidence',function(mount){
    var q=A.slider({label:'Belief q = Q(on)',min:0,max:1,step:0.001,value:0.5,decimals:3});
    var theta=A.slider({label:'Model θ = P(bright | on)',min:0.1,max:0.95,step:0.01,value:0.8});
    var out=A.readout(),body=el('div');
    function render(){
      var b=q.get(),j=[0.5*theta.get(),0.1],ev=j[0]+j[1],post=j[0]/ev;
      var F=(b>0?b*Math.log(b/j[0]):0)+(b<1?(1-b)*Math.log((1-b)/j[1]):0),surprise=-Math.log(ev);
      out.show([['Evidence',ev.toFixed(4)],['Surprise (nats)',surprise.toFixed(4)],['Free energy (nats)',F.toFixed(4)],['KL gap (nats)',Math.max(0,F-surprise).toFixed(4)]]);
      body.replaceChildren(table(['State','Approximate belief','Exact posterior'],[['on',b.toFixed(3),post.toFixed(3)],['off',(1-b).toFixed(3),(1-post).toFixed(3)]]));
    }
    mount.appendChild(A.panel([A.row([q.el,theta.el]),A.button('Match posterior',function(){q.set(theta.get()/(theta.get()+0.2));}),out,body,A.note('The observation is fixed as bright; P(on) is 0.50 and P(bright | off) is 0.20. Changing q is inference. Changing θ changes the model. The button matches the posterior to slider resolution.')]));
    q.onchange(render);theta.onchange(render);render();
  });
  A.register('inference-comparison',function(mount){
    var count=A.slider({label:'Posterior samples',min:20,max:5000,step:20,value:500,decimals:0});
    var body=el('div'),out=A.readout(),seed=17;
    var prior=[0.81,0.09,0.09,0.01],lik=[0.01,0.901,0.901,0.9901];
    var j=prior.map(function(p,i){return p*lik[i];}),z=j.reduce(function(a,b){return a+b;},0),P=j.map(function(x){return x/z;});
    var q1=0.2,q2=0.8;
    function sigmoid(x){return 1/(1+Math.exp(-x));}
    for(var k=0;k<400;k++){
      q1=sigmoid((1-q2)*Math.log(P[2]/P[0])+q2*Math.log(P[3]/P[1]));
      q2=sigmoid((1-q1)*Math.log(P[1]/P[0])+q1*Math.log(P[3]/P[2]));
    }
    var Q=[(1-q1)*(1-q2),(1-q1)*q2,q1*(1-q2),q1*q2];
    function kl(q){return q.reduce(function(a,v,i){return a+(v>0?v*Math.log(v/P[i]):0);},0);}
    function render(){
      var rng=seed,c=[0,0,0,0],N=count.get();
      for(var a=0;a<N;a++){rng=(Math.imul(1664525,rng)+1013904223)>>>0;var u=rng/4294967296,i=0,cum=P[0];while(i<3&&u>cum){i++;cum+=P[i];}c[i]++;}
      var empirical=c.map(function(x){return x/N;});
      body.replaceChildren(table(['Gust, baboon','Exact','Empirical','Mean-field'],['0, 0','0, 1','1, 0','1, 1'].map(function(label,i){return [label,P[i].toFixed(4),empirical[i].toFixed(4),Q[i].toFixed(4)];})));
      out.show([['Seed',seed],['Empirical KL (nats)',kl(empirical).toFixed(4)],['Mean-field KL (nats)',kl(Q).toFixed(4)]]);
    }
    mount.appendChild(A.panel([count.el,A.button('Repeat draws',function(){seed++;render();}),out,body,A.note('Draws use the exact categorical posterior. This isolates sampling variation from factorisation error and does not measure MCMC runtime. Mean-field uses one asymmetric initialisation; swapping the causes gives the symmetric alternative solution.')]));
    count.onchange(render);render();
  });
  A.register('posterior-decisions',function(mount){
    var p=A.slider({label:'Shared posterior hazard probability',min:0,max:1,step:0.01,value:0.3});
    var body=el('div');
    function render(){
      var stay=10*p.get();
      body.replaceChildren(table(['Animal','Loss if staying','Loss if leaving','Preferred action'],[2,5].map(function(cost,i){return [i===0?'A':'B',stay.toFixed(2),cost.toFixed(2),Math.abs(stay-cost)<1e-10?'tie':(cost<stay?'leave':'stay')];})));
    }
    mount.appendChild(A.panel([p.el,body,A.note('Staying costs 10 if the hazard is present and zero otherwise. Leaving costs 2 for A and 5 for B, regardless of state. Losses use a common arbitrary unit.')]));p.onchange(render);render();
  });
})();
