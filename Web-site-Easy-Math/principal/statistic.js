
function calculate() {
    const numElements = parseInt(document.getElementById("numElements").value);
    const elementInputs = document.getElementById("elementInputs");
    const cis = [];

    for (let i = 1; i <= numElements; i++) {
        const input = parseFloat(document.getElementById(`element${i}`).value);
        cis.push(input);
    }
    document.getElementById("result").innerHTML = "Multiplied Array: " + cis.join(", ");
   
  
}




function calculates() {
    const numElements = parseInt(document.getElementById("numElements").value);
    const eI = document.getElementById("eI");
    const nis = [];

    for (let i = 1; i <= numElements; i++) {
        const input = parseFloat(document.getElementById(`element0${i}`).value);
        nis.push(input);
    }
    document.getElementById("results").innerHTML = "Multiplied Array: " + nis.join(", ");
   
  
}


function multiplyArray(){

    //supposons les modalites ci les effectifs ni




    

    var T=0;
    for(var i=0 , N=cis.length ; i<N ; i++){ // calcule de la somme des effectifs
        T= T + parseFloat(nis[i]);
    }

    
     
  

   
   
    var freq=nis.map(function(number){return (number*100)/T;}); // calcul des frequences
    var effdegret=nis.map(function(number){return (number*360)/T;}); // cacul des angles
    var nicumule=[];
    var freqcumule=[];



    var h=0;
    for(var i=0,  N=cis.length; i<N;i++){// remplissage des effectifs cumulés croissants
        h=h + parseFloat(nis[i]);
         nicumule.push(h);
     }

    var h=0;
    for(var i=0, N=cis.length; i<N;i++){//remplissage des frequence cumulés croissant 
        h=h + parseFloat(freq[i]);
         freqcumule.push(h);
     }

    var nici=[];
    for(var i=0,  N=cis.length; i<N;i++){// remplissage du tableau des ni*ci
       
         nici.push(parseFloat(cis[i]*nis[i]));
     }


     var nicici=[];
    for(var i=0,  N=cis.length; i<N;i++){// remplissage du tableau des ni*ci*ci
       
         nicici.push(parseFloat(cis[i]*nis[i]*cis[i]));
     }


     var k1=0;
     for(var i=0,  N=cis.length; i<N;i++){//  somme des nici
         k1=k1 + parseFloat(nici[i]);
      }
      var moy=k1/T;// moyenne des nici

      var k2=0;
      for(var i=0,  N=cis.length; i<N;i++){//  somme des nicici
        k2=k2 + parseFloat(nicici[i]);
     }

     var variance=(k2/T) -(moy*moy);// calcule de la variance
     var ec=Math.sqrt(variance);// calcule de l'ecart type






    document.getElementById("T").innerHTML=T;
    document.getElementById("nicumule").innerHTML=nicumule.join("  ;   ");
    document.getElementById("freq").innerHTML=freq.join("  ;  ");
    document.getElementById("freqcumule").innerHTML=freqcumule.join("  ;  ");
    document.getElementById("effdegret").innerHTML=effdegret.join("  ;  ");
    document.getElementById("nici").innerHTML=nici.join("  ;  ");
    document.getElementById("k1").innerHTML=k1;
    document.getElementById("moy").innerHTML=moy;
    document.getElementById("nicici").innerHTML=nicici.join("  ;  ");
    document.getElementById("k2").innerHTML=k2;
    document.getElementById("variance").innerHTML=variance;
    document.getElementById("ec").innerHTML=ec;


   
    
}

















function intervalle(){

    //supposons les modalites ci les effectifs ni


    var modalite1Text=document.getElementById("modalite0").value;  //get the numbers from the textarea 
    var modalite2Text=document.getElementById("modalite1").value; 
    var numbers1Text=document.getElementById("effectifss").value;

    var cis0= modalite1Text.split("\n");// Split the numbers into an array 
    var cis1= modalite2Text.split("\n");
    var niss=numbers1Text.split("\n");// transformation des elmts entrer sous forme de tableau
    
    var ciss=[];

    for(var i=0,N=cis0.length ; i<N; i++){
        ciss.push(  (parseFloat(cis0[i] + cis1[i])) /2 );
    }

    var Ts=0;
    for(var i=0 , N=ciss.length ; i<N ; i++){ // calcule de la somme des effectifs
        Ts= Ts + parseFloat(niss[i]);
    }

    
     
  

   
   
    var freqs=niss.map(function(number){return (number*100)/Ts;}); // calcul des frequences
    var effdegrets=niss.map(function(number){return (number*360)/Ts;}); // cacul des angles
    var nicumules=[];
    var freqcumules=[];



    var h=0;
    for(var i=0,  N=ciss.length; i<N;i++){// remplissage des effectifs cumulés croissants
        h=h + parseFloat(niss[i]);
         nicumules.push(h);
     }

    var h=0;
    for(var i=0, N=ciss.length; i<N;i++){//remplissage des frequence cumulés croissant 
        h=h + parseFloat(freqs[i]);
         freqcumules.push(h);
     }

    var nicis=[];
    for(var i=0,  N=ciss.length; i<N;i++){// remplissage du tableau des ni*ci
       
         nicis.push(parseFloat(ciss[i]*niss[i]));
     }


     var nicicis=[];
    for(var i=0,  N=ciss.length; i<N;i++){// remplissage du tableau des ni*ci*ci
       
         nicicis.push(parseFloat(ciss[i]*niss[i]*ciss[i]));
     }


     var k1s=0;
     for(var i=0,  N=ciss.length; i<N;i++){//  somme des nici
         k1s=k1s + parseFloat(nicis[i]);
      }
      var moys=k1s/Ts;// moyenne des nici

      var k2s=0;
      for(var i=0,  N=ciss.length; i<N;i++){//  somme des nicici
        k2s=k2s + parseFloat(nicicis[i]);
     }

     var variances=(k2s/Ts) -(moys*moys);// calcule de la variance
     var ecs=Math.sqrt(variances);// calcule de l'ecart type






    document.getElementById("Ts").innerHTML=Ts;
    document.getElementById("nicumules").innerHTML=nicumules.join("  ;   ");
    document.getElementById("freqs").innerHTML=freqs.join("  ;  ");
    document.getElementById("freqcumules").innerHTML=freqcumules.join("  ;  ");
    document.getElementById("effdegrets").innerHTML=effdegrets.join("  ;  ");
    document.getElementById("nicis").innerHTML=nicis.join("  ;  ");
    document.getElementById("k1s").innerHTML=k1s;
    document.getElementById("moys").innerHTML=moys;
    document.getElementById("nicicis").innerHTML=nicicis.join("  ;  ");
    document.getElementById("k2s").innerHTML=k2s;
    document.getElementById("variances").innerHTML=variances;
    document.getElementById("ecs").innerHTML=ecs;


   
    
}



function calculate() {
    const numElements = parseInt(document.getElementById("numElements").value);
    const elementInputs = document.getElementById("elementInputs");
    const elements = [];

    for (let i = 1; i <= numElements; i++) {
        const input = parseFloat(document.getElementById(`element${i}`).value);
        elements.push(input);
    }

    const multipliedArray = elements.map(element => element * 2);

    document.getElementById("result").innerHTML = "Multiplied Array: " + multipliedArray.join(", ");
}