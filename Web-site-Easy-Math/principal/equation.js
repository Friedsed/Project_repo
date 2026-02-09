

function find_equation(){
    var xa= parseFloat(document.getElementById("xa").value);
    var ya= parseFloat(document.getElementById("ya").value);
    var xb= parseFloat(document.getElementById("xb").value);
    var yb= parseFloat(document.getElementById("yb").value);
    
    var bb=-(xb-xa);
    var aa=yb-ya ;
    var cc=-aa*xa-bb*ya;
    var equation='l\'equation est :' + aa +' x + (' + bb + ') y + ( ' + cc +')= 0 ';
    document.getElementById("equation").innerHTML=equation;
}



function find_resolution(){
    var a= parseFloat(document.getElementById("a").value);
    var b= parseFloat(document.getElementById("b").value);
    var c= parseFloat(document.getElementById("c").value);
    var disc = b*b-4*a*c;
    if(disc > 0){
        var x=(-b+Math.sqrt(disc))/2*a;
        var x1=(-b-Math.sqrt(disc))/2*a;
        var result='On a deux solutions : x1=' + x + 'et x2 = ' + x1 ;
    }
    if(disc < 0){
        var x=(-b/2*a);
        var x1=Math.sqrt(-disc)/2*a;
        var result='On a deux solutions complexes: x1= ' + x + ' + (' + x1 + ') i  et x2= ' + x + ' - ( ' + x1 +') i' ;
    }
    if(disc===0){
        var x=-b/2*a;
        var result='On a une solution double : x= ' + x;
    }
    document.getElementById("result").innerHTML=result;
}




function find_solution(){
    var k1= parseFloat(document.getElementById("k1").value);
    var k2= parseFloat(document.getElementById("k2").value);
    var k3= parseFloat(document.getElementById("k3").value);
    var k4= parseFloat(document.getElementById("k4").value);
    var k5= parseFloat(document.getElementById("k5").value);
    var k6= parseFloat(document.getElementById("k6").value);
    if(k1<0){
        k1=-k1;
        k2=-k2;
        k3=-k3;
    }
    if(k4<0){
        k4=-k4;
        k5=-k5;
        k6=-k6;
    }
    var y=(k4*k3-k1*k6)/(k4*k2-k1*k5);
    var x=(-y*k2+k3)/k1 ;
    var resultat='la solution est : x= '+ x+'; y = '+ y;
    document.getElementById("resultat").innerHTML=resultat;
}



function find_solutiont(){
    var a0= parseFloat(document.getElementById("a0").value);
    var b0= parseFloat(document.getElementById("b0").value);
    var c0= parseFloat(document.getElementById("c0").value);
    var d0= parseFloat(document.getElementById("d0").value);
    var a1= parseFloat(document.getElementById("a1").value);
    var b1= parseFloat(document.getElementById("b1").value);
    var c1= parseFloat(document.getElementById("c1").value);
    var d1= parseFloat(document.getElementById("d1").value);
    var a2= parseFloat(document.getElementById("a2").value);
    var b2= parseFloat(document.getElementById("b2").value);
    var c2= parseFloat(document.getElementById("c2").value);
    var d2= parseFloat(document.getElementById("d2").value);
   


    
        if(a0>0)
        {
        a0=-a0;
        b0=-b0;
        c0=-c0;
        d0=-d0;

        }
        if(a1>0)
        {
        a1=-a1;
        b1=-b1;
        c1=-c1;
        d1=-d1;

        }
        if(a2>0)
        {
        a2=-a2;
        b2=-b2;
        c2=-c2;
        d2=-d2;

        }


        var k7=-b1*a0+b0*a1;
        var k8=-c1*a0+c0*a1;
        var k9=-d1*a0+d0*a1;
        var k10=-b2*a0+b0*a2;
        var k11=-c2*a0+c0*a2;
        var k12=-d2*a0+d0*a2;
        if (k7<0)
        {
        k7=-k7;
        k8=-k8;
        k9=-k9;
        }

        if(k10<0)
        {
        k10=-k10;
        k11=-k11;
        k12=-k12;
        }




        var z=(k10*k9-k7*k12)/(k10*k8-k7*k11);
        var y1=(-z*k8+k9)/k7;
        var x1=-(b0*y1+c0*z+d0)/a0;
        var resultatt='La solution est de la forme (x,y,z) : '+' ( '+ x1 +' ; ' + y1+ ' ; ' + z +')' ;

        document.getElementById("resultatt").innerHTML=resultatt;
       

}