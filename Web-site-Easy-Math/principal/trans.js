




class Point2D{
    constructor(abs,ord){
        this.abs=abs;
        this.ord=ord;
    }
    
    vecteur2D(other){
        return new Point2D(other.abs-this.abs,other.ord-this.ord);
    }
     // on suppose la translation de vecteur other et le point M pour coordonnée this abs et thus ord

    translation2D(other){
        return new Point2D(other.abs+this.abs,other.ord+this.ord);
    }
    // on suppose l'homothetie de centre other et le point M pour coordonnée this abs et thus ord

    homothetie2D(other,k){
        return new Point2D(k*this.abs+other.abs*(1-k),this.ord*k+other.ord*(1-k));

    }
     // on suppose une equation de droite de la forme ax+by+c=0 d'ou les valeurs de a et b utilisé
   

  
    toString2D(){
            return '(' + this.abs + '  ,  ' + this.ord +')';
        }

    
}






class Point3D{
    constructor(abs,ord,cot){
        this.abs=abs;
        this.ord=ord;
        this.cot=cot;
    }
    // vecteur3D veut dire vecteur trois D
    // calcule de produit vectoriel de vecteur de l'expace
    // supposons AB vectoriel AC c'est AC qui représente other
    pvectoriel3D(other){
        var n1=this.ord*other.cot - other.ord*this.cot;
        var n2=this.cot*other.abs - other.cot*this.abs;
        var n3=this.abs*other.ord - other.abs*this.ord;
        return new Point3D(n1, n2, n3);
    }
    vecteur3D(other){
        return new Point3D(other.abs-this.abs,other.ord-this.ord,other.cot-this.cot);
    }
    // on suppose la translation de vecteur other et le point M pour coordonnée this abs et thus ord

    translation3D(other){
        return new Point3D(other.abs+this.abs, other.ord+this.ord, other.cot+this.cot);
    }
    // on suppose l'homothetie de centre other et le point M pour coordonnée this abs et thus ord
    
    homothetie3D(other,k){
        return new Point3D(k*this.abs+other.abs*(1-k), this.ord*k+other.ord*(1-k), this.cot*k+other.cot*(1-k) );
        
    }
    // on suppose une equation de plan de la forme ax+by+cz+d=0 d'ou les valeurs de a et b utilisé
    projection3D(a,b,c,d){
        var k=-(this.abs*a+this.ord*b+this.cot*c+d)/(a*a+b*b+c*c);
        var a1 = this.abs+k*a;
        var b1= this.ord+k*b;
        var c1= this.cot+k*c;
        return new Point3D(a1,b1,c1);

    // on considère tj la projection orthoganal par rapport a la plan d'equ ax+by+cz+d=0
    }
    reflexion3D(a,b,c,d){
        var k=-(this.abs*a+this.ord*b+this.cot*c+d)/(a*a+b*b+c*c);
        var a1 = this.abs+k*a;
        var b1= this.ord+k*b;
        var c1= this.cot+k*c;
        return new Point3D(2*a1-this.abs , 2*b1-this.ord , 2*c1-this.cot);
        
    }
    
    sol3i(b){
        if(this.abs<0){
            this.abs=-this.abs;
            this.ord=-this.ord;
            this.cot=-this.cot;
        }
        if(b.abs<0){
            b.abs=-b.abs;
            b.ord=-b.ord;
            b.cot=-b.cot;
        }
        var y=(-b.abs*this.cot+this.abs*b.cot)/(b.abs*this.ord-this.abs*b.ord);
        var x=(-y*this.ord-this.cot)/this.abs ;
        return new Point3D( x , y , 0);
        
        
    }// "b" et "this" remplacent les coordonées de l'equation 
    // c et b les x et y du point M
    sym_orth(b,c,d){
        if(this.abs<0){
            this.abs=-this.abs;
            this.ord=-this.ord;
            this.cot=-this.cot;
        }
        if(b.abs<0){
            b.abs=-b.abs;
            b.ord=-b.ord;
            b.cot=-b.cot;
        }
        var y=(-b.abs*this.cot+this.abs*b.cot)/(b.abs*this.ord-this.abs*b.ord);
        var x=(-y*this.ord-this.cot)/this.abs ;
        return new Point3D(2*x - c , 2*y - d , 0);
       
    }
    
    
        toString3D(){
            return '(' + this.abs + '  ,  ' + this.ord+ '  ,  ' + this.cot +')';
        }
        
        
    }
    



    
class Point4D{
    constructor(abs,ord,cot,li){
        this.abs=abs;
        this.ord=ord;
        this.cot=cot;
        this.li=li;
    }
    sol4i(a,b){

        if(this.abs<0){
            this.abs=-this.abs;
            this.ord=-this.ord;
            this.cot=-this.cot;
            this.li=-this.li;
        }
        if(a.abs<0){
            a.abs=-a.abs;
            a.ord=-a.ord;
            a.cot=-a.cot;
            a.li=-a.li;
        }
        if(b.abs<0){
            b.abs=-b.abs;
            b.ord=-b.ord;
            b.cot=-b.cot;
            b.li=-b.li;
        }


        var k7= -a.ord*this.abs + this.ord*a.abs;
        var k8= -a.cot*this.abs + this.cot*a.abs;
        var k9= -a.li*this.abs + this.li*a.abs;
        var k10= -b.ord*this.abs + this.ord*b.abs;
        var k11= -b.cot*this.abs + this.cot*b.abs;
        var k12= -b.li*this.abs + this.li*b.abs;

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
        var y=(-z*k8+k9)/k7;
        var x= -(this.ord * y + this.cot * z - this.li ) / this.abs ;
       
        return new Point4D( x , y , z,0);
    }



    toString4D(){
        return '(' + this.abs + ' ; ' + this.ord+ ' ; ' + this.cot + ' ; '+ 0 + ')'    ;
    }



}





    function distance2D(a,b){
        return Math.sqrt( Math.pow(a.abs-b.abs,2)+Math.pow(a.ord-b.ord,2));
    }
    
    function distance3D (a,b) {
        return Math.sqrt( Math.pow(a.abs-b.abs,2)+Math.pow(a.ord-b.ord,2)+Math.pow(a.cot-b.cot,2));
    }
    

    function pscalaire(a,b){
        return (a.abs*b.abs+a.ord*b.ord+a.cot*b.cot);
    }





















    // Les fonctions de verification en 3 Dimensions








    function find_vector(){
        var e4=parseFloat(document.getElementById("e4").value);
        var e5=parseFloat(document.getElementById("e5").value);
        var e6=parseFloat(document.getElementById("e6").value);
        var e7=parseFloat(document.getElementById("e7").value);
        var f2= new Point2D(e4,e5);
        var f3= new Point2D(e6,e7);
        var vector2D= f2.vecteur2D(f3);
    
        document.getElementById("vector2D").innerHTML= vector2D.toString2D();
    
    }





    function find_translation(){
        var e8=parseFloat(document.getElementById("e8").value);
        var e9=parseFloat(document.getElementById("e9").value);
        var e10=parseFloat(document.getElementById("e10").value);
        var e11=parseFloat(document.getElementById("e11").value);
        var f4= new Point2D(e8,e9);
        var f5= new Point2D(e10,e11);
        var trans2D= f4.translation2D(f5);
    
        document.getElementById("trans2D").innerHTML= trans2D.toString2D();
    
    }
    
    
    function find_homothety(){
        var e12=parseFloat(document.getElementById("e12").value);
        var e13=parseFloat(document.getElementById("e13").value);
        var e14=parseFloat(document.getElementById("e14").value);
        var e15=parseFloat(document.getElementById("e15").value);
        var e16=parseFloat(document.getElementById("e16").value);
        var f6= new Point2D(e12,e13);
        var f7= new Point2D(e14,e15);
        var homo2D= f6.homothetie2D(f7,e16);
    
        document.getElementById("homo2D").innerHTML= homo2D.toString2D();
    
    }



// e17 e18 sont les x et y du point M 
// e19 e20 e21 sont les a b et c de l'equation de doite sur laquelle se trouve le projeter ax+by+c=0
    function find_projection(){
        var e17=parseFloat(document.getElementById("e17").value);
        var e18=parseFloat(document.getElementById("e18").value);
        var e19=parseFloat(document.getElementById("e19").value);
        var e20=parseFloat(document.getElementById("e20").value);
        var e21=parseFloat(document.getElementById("e21").value);
    
            var f20= e17*e20 - e18 * e19;
            var f21= new Point3D(e19,e20,e21);
            var f22= new Point3D(-e20,e19,f20);
            var proj2D=f21.sol3i(f22);
           
        
            // var proj2D= new Point2D (f23.abs , f23.ord);  
    
        document.getElementById("proj2D").innerHTML= proj2D.toString3D();
    
    }



    // e22 et e23 remplacent les x et y du point M
    // e24 e25 e26  remplacent les a b et c de l'equation de droite ax+by+c=0

    function find_symmetry_orthogonal(){
        var e22=parseFloat(document.getElementById("e22").value);
        var e23=parseFloat(document.getElementById("e23").value);
        var e24=parseFloat(document.getElementById("e24").value);
        var e25=parseFloat(document.getElementById("e25").value);
        var e26=parseFloat(document.getElementById("e26").value);
        var f24= e22*e25 - e23 * e24;
            var f25= new Point3D(e24,e25,e26);
            var f26= new Point3D(-e25,e24,f24);
            var so=f25.sym_orth(f26,e22,e23);
           
    
        document.getElementById("so").innerHTML= so.toString3D();
    
    }






    function find_dotProduct3D(){
        var n7=parseFloat(document.getElementById("n7").value);
        var n8=parseFloat(document.getElementById("n8").value);
        var n9=parseFloat(document.getElementById("n9").value);
        var n10=parseFloat(document.getElementById("n10").value);
        var n11=parseFloat(document.getElementById("n11").value);
        var n12=parseFloat(document.getElementById("n12").value);
        var t2= new Point3D(n7,n8,n9);
        var t3= new Point3D(n10,n11,n12);
        var pscalaire=pscalaire(t2,t3);
    
    document.getElementById("pscalaire").innerHTML= pscalaire;
    
        
    }
    


    function find_crossProduct3D(){
        var n1=parseFloat(document.getElementById("n1").value);
        var n2=parseFloat(document.getElementById("n2").value);
        var n3=parseFloat(document.getElementById("n3").value);
        var n4=parseFloat(document.getElementById("n4").value);
        var n5=parseFloat(document.getElementById("n5").value);
        var n6=parseFloat(document.getElementById("n6").value);
        var t0= new Point3D(n1,n2,n3);
        var t1= new Point3D(n4,n5,n6);
        var pvectoriel= t0.pvectoriel3D(t1);
    
    document.getElementById("pvectoriel").innerHTML= pvectoriel.toString3D();
    
        
    }

    

    
function find_distance3D(){
    var n26=parseFloat(document.getElementById("n26").value);
    var n27=parseFloat(document.getElementById("n27").value);
    var n28=parseFloat(document.getElementById("n28").value);
    var n29=parseFloat(document.getElementById("n29").value);
    var n30=parseFloat(document.getElementById("n30").value);
    var n31=parseFloat(document.getElementById("n31").value);
    var t8= new Point3D(n26,n27,n28);
    var t9= new Point3D(n29,n30,n31);
    var dist3D= distance3D(t8,t9);

document.getElementById("dist3D").innerHTML= dist3D;


}




function find_point_plane3D(){
    var n32=parseFloat(document.getElementById("n32").value);
    var n33=parseFloat(document.getElementById("n33").value);
    var n34=parseFloat(document.getElementById("n34").value);
    var n35=parseFloat(document.getElementById("n35").value);
    var n36=parseFloat(document.getElementById("n36").value);
    var n37=parseFloat(document.getElementById("n37").value);
    var n38=parseFloat(document.getElementById("n38").value);
    var t10= new Point3D(n32,n33,n34);// coordone=ée de M
    var t11= t10.projection3D(n35,n36,n37,n38);// coordornée de H projeté de M sur le plan
    var dpp=distance3D(t11,t10);// distance Mh

    document.getElementById("dpp").innerHTML= dpp;

}



function find_vector3D(){
    var n39=parseFloat(document.getElementById("n39").value);
    var n40=parseFloat(document.getElementById("n40").value);
    var n41=parseFloat(document.getElementById("n41").value);
    var n42=parseFloat(document.getElementById("n42").value);
    var n43=parseFloat(document.getElementById("n43").value);
    var n44=parseFloat(document.getElementById("n44").value);
    var t12= new Point3D(n39,n40,n41);
    var t13= new Point3D(n42,n43,n44);
    var vecteur3D= t12.vecteur3D(t13);

    document.getElementById("vecteur3D").innerHTML= vecteur3D.toString3D();

}
function find_plane_equation3D(){
    var n95=parseFloat(document.getElementById("n95").value);
    var n96=parseFloat(document.getElementById("n96").value);
    var n97=parseFloat(document.getElementById("n97").value);
    var n98=parseFloat(document.getElementById("n98").value);
    var n99=parseFloat(document.getElementById("n99").value);
    var n100=parseFloat(document.getElementById("n100").value);
    var n101=parseFloat(document.getElementById("n101").value);
    var n102=parseFloat(document.getElementById("n102").value);
    var n103=parseFloat(document.getElementById("n103").value);

    var t30= new Point3D(n95,n96,n97);
    var t31= new Point3D(n98,n99,n100);
    var t32= new Point3D(n101,n102,n103);
    var t33= t30.vecteur3D(t31);
    var t34= t30.vecteur3D(t32);
    var t35= t33.pvectoriel3D(t34);
    var t36=t35.abs;
    var t37=t35.ord;
    var t38=t35.cot;
    var t39=-(n95*t36+n96*t37+n97*t38);
    document.getElementById("t39").innerHTML= '( '+t36+ ' ) x + '+' ('+ t37 + ') y '+ '+('+ t38+ ') z + '+ t39 +' = 0';

}


function find_translation3D(){
    var n45=parseFloat(document.getElementById("n45").value);
    var n46=parseFloat(document.getElementById("n46").value);
    var n47=parseFloat(document.getElementById("n47").value);
    var n48=parseFloat(document.getElementById("n48").value);
    var n49=parseFloat(document.getElementById("n49").value);
    var n50=parseFloat(document.getElementById("n50").value);
    var t14= new Point3D(n45,n46,n47);
    var t15= new Point3D(n48,n49,n50);
    var trans3D= t14.translation3D(t15);

    document.getElementById("trans3D").innerHTML= trans3D.toString3D();

}



function find_homothety3D(){
    var n51=parseFloat(document.getElementById("n51").value);
    var n52=parseFloat(document.getElementById("n52").value);
    var n53=parseFloat(document.getElementById("n53").value);
    var n54=parseFloat(document.getElementById("n54").value);
    var n55=parseFloat(document.getElementById("n55").value);
    var n56=parseFloat(document.getElementById("n56").value);
    var n57=parseFloat(document.getElementById("n57").value);
    var t16= new Point3D(n51,n52,n53);
    var t17= new Point3D(n54,n55,n56);
    var homo3D= t16.homothetie3D(t17,n57);

    document.getElementById("homo3D").innerHTML= homo3D.toString3D();

}



function find_projection3D(){
    var n58=parseFloat(document.getElementById("n58").value);
    var n59=parseFloat(document.getElementById("n59").value);
    var n60=parseFloat(document.getElementById("n60").value);
    var n61=parseFloat(document.getElementById("n61").value);
    var n62=parseFloat(document.getElementById("n62").value);
    var n63=parseFloat(document.getElementById("n63").value);
    var n64=parseFloat(document.getElementById("n64").value);
    var t18= new Point3D(n58,n59,n60);
    var proj3D= t18.projection3D(n61,n62,n63,n64);

    document.getElementById("proj3D").innerHTML= proj3D.toString3D();

}




function find_reflection3D(){
    var n65=parseFloat(document.getElementById("n65").value);
    var n66=parseFloat(document.getElementById("n66").value);
    var n67=parseFloat(document.getElementById("n67").value);
    var n68=parseFloat(document.getElementById("n68").value);
    var n69=parseFloat(document.getElementById("n69").value);
    var n70=parseFloat(document.getElementById("n70").value);
    var n71=parseFloat(document.getElementById("n71").value);
    var t19= new Point3D(n65,n66,n67);
    var refl3D= t19.reflexion3D(n68,n69,n70,n71);

    document.getElementById("refl3D").innerHTML= refl3D.toString3D();

}



function find_quadratic_equation(){
    var n115= parseFloat(document.getElementById("n115").value);
    var n116= parseFloat(document.getElementById("n116").value);
    var n117= parseFloat(document.getElementById("n117").value);
    var disc = n116*n116-4*n115*n117;
    if(disc > 0){
        var t40=(-n116+Math.sqrt(disc))/2*n115;
        var t41=(-n116-Math.sqrt(disc))/2*n115;
        var res='On a deux solutions : x1=' + t40 + 'et x2 = ' + t41 ;
    }
    else if(disc < 0){
        var t40=(-n116/2*n115);
        var t41=Math.sqrt(-disc)/2*n115;
        var res='On a deux solutions complexes: x1= ' + t40 + ' + (' + t41 + ') i  et x2= ' + t40 + ' - ( ' + t41 +') i' ;
    }
    else if(disc===0){
        var t40=-n116/2*n115;
        var res='On a une solution double : x= ' + t40;
    }
    document.getElementById("res").innerHTML=res;
}



function find_solution2I(){
    var n20=parseFloat(document.getElementById("n20").value);
    var n21=parseFloat(document.getElementById("n21").value);
    var n22=parseFloat(document.getElementById("n22").value);
    var n23=parseFloat(document.getElementById("n23").value);
    var n24=parseFloat(document.getElementById("n24").value);
    var n25=parseFloat(document.getElementById("n25").value);
    var p20= new Point3D(n20,n21,n22);
    var p21= new Point3D(n23,n24,n25);
    var p22= new Point3D(-n21,n20,0);
    var p23= new Point3D(n23,n24,0);
    
    if (pscalaire(p22,p23)===0){
        var p24= n20+ ' x + '+ n21 +' y + '+  n22 + ' = 0 ';
        document.getElementById("p24").innerHTML= p24;
    }
    else{

        var p24 =p20.sol3i(p21);
        document.getElementById("p24").innerHTML= p24.toString3D();
   }

}




function find_solution3I(){
    var n72=parseFloat(document.getElementById("n72").value);
    var n73=parseFloat(document.getElementById("n73").value);
    var n74=parseFloat(document.getElementById("n74").value);
    var n75=parseFloat(document.getElementById("n75").value);
    var n76=parseFloat(document.getElementById("n76").value);
    var n77=parseFloat(document.getElementById("n77").value);
    var n78=parseFloat(document.getElementById("n78").value);
    var n79=parseFloat(document.getElementById("n79").value);
    var n80=parseFloat(document.getElementById("n80").value);
    var n81=parseFloat(document.getElementById("n81").value);
    var n82=parseFloat(document.getElementById("n82").value);
    var n83=parseFloat(document.getElementById("n83").value);

    var t20= new Point4D(n72,n73,n74,n75);
    var t21= new Point4D(n76,n77,n78,n79);
    var t22= new Point4D(n80,n81,n82,n83);

    var t23= new Point3D(n72,n73,n74);
    var t24= new Point3D(n76,n77,n78);
    var t25= new Point3D(n80,n81,n82);

   
    
    if (t23.pvectoriel3D(t24).abs===0 && t23.pvectoriel3D(t24).ord===0 && t23.pvectoriel3D(t24).cot===0 ){
        var t26= n72+ ' x + '+ n73 +' y + '+  n74 + ' z + ' + n75 +' = 0 ';
        var t27= n80+ ' x + '+ n81 +' y + '+  n82 + ' z + ' + n83 +' = 0 ';
        document.getElementById("t26").innerHTML= t26;
        document.getElementById("t27").innerHTML= t27;
    }
    else if (t24.pvectoriel3D(t25).abs===0 && t24.pvectoriel3D(t25).ord===0 && t24.pvectoriel3D(t25).cot===0  ){
        var t26= n72+ ' x + '+ n73 +' y + '+  n74 + ' z + ' + n75 +' = 0 ';
        var t27= n80+ ' x + '+ n81 +' y + '+  n82 + ' z + ' + n83 +' = 0 ';
        document.getElementById("t26").innerHTML= t26;
        document.getElementById("t27").innerHTML= t27;
    }
    else if (t23.pvectoriel3D(t25).abs===0 && t23.pvectoriel3D(t25).ord===0 && t23.pvectoriel3D(t25).cot===0 ){
        var t26= n80+ ' x + '+ n81 +' y + '+  n82 + ' z + ' + n83 +' = 0 ';
        var t27= n76+ ' x + '+ n77 +' y + '+  n78 + ' z + ' + n79 +' = 0 ';

        document.getElementById("t26").innerHTML= t26;
        document.getElementById("t27").innerHTML= t27;
    }


    else{

        var t28=t20.sol4i(t21,t22);
        document.getElementById("t28").innerHTML= t28.toString4D();
   }

}




function solveSystem() {
    const equations = [];
    const numEquations = parseInt(document.getElementById("numEquations").value);

    for (let i = 1; i <= numEquations; i++) {
        const coefficients = [];
        for (let j = 1; j <= numEquations + 1; j++) {
            coefficients.push(parseFloat(document.getElementById(`coeff${i}${j}`).value));
        }
        equations.push(coefficients);
    }
    document.getElementById("equations").innerHTML=equations.join(" ; ");
    // for (let i = 0; i < numEquations; i++) {
    //     for (let j = i + 1; j < numEquations; j++) {
    //         const factor = equations[j][i] / equations[i][i];
    //         for (let k = i; k <= numEquations; k++) {
    //             equations[j][k] -= factor * equations[i][k];
    //         }
    //     }
    // }

    // const solutions = [];
    // for (let i = numEquations - 1; i >= 0; i--) {
    //     let sum = equations[i][numEquations];
    //     for (let j = i + 1; j < numEquations; j++) {
    //         sum -= equations[i][j] * solutions[j];
    //     }
    //     solutions[i] = sum / equations[i][i];
    // }

    // const solutionOutput = document.getElementById("solution");
    // solutionOutput.innerHTML = "Solutions: " + solutions.join(", ");
}



