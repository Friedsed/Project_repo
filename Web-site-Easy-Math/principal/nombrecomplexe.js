class ComplexNumber{
    constructor(real,imag){
        this.real=real;
        this.imag=imag;    }

        add(other){
            return new ComplexNumber(this.real+other.real,this.imag+other.imag);
        }
        times(other){
            return new ComplexNumber(this.real*other.real-this.imag*other.imag , this.imag*other.real+this.real*other.imag);
        }
        divided(other){
            var a=Math.sqrt(other.real*other.real + other.imag*other.imag);
            return new ComplexNumber((this.real*other.real+this.imag*other.imag)/a , (this.imag*other.real-this.real*other.imag)/a );
        }
        toString(){
            return this.real + ' + '+ this.imag +' i';
        }
}
function module(a){
    return Math.sqrt(a.real*a.real + a.imag*a.imag);
}

function argument(a){
    return Math.acos(a.real/module(a));
}


function find_module(){
    var rel=parseFloat(document.getElementById("rel").value);
    var img=parseFloat(document.getElementById("img").value);
    var c1= new ComplexNumber(rel,img);

    var modul = module(c1);
    document.getElementById("modul").innerHTML=modul;
}


function find_argument(){
    var rel1=parseFloat(document.getElementById("rel1").value);
    var img1=parseFloat(document.getElementById("img1").value);
    var c2= new ComplexNumber(rel1,img1);

    var arg= argument(c2);
    var degre=(arg*180)/(22/7);

    document.getElementById("arg").innerHTML=arg;
    document.getElementById("degre").innerHTML=degre;

}

function sommecomplexe(){
    var rel2=parseFloat(document.getElementById("rel2").value);
    var img2=parseFloat(document.getElementById("img2").value);
    var rel3=parseFloat(document.getElementById("rel3").value);
    var img3=parseFloat(document.getElementById("img3").value);

    var c3= new ComplexNumber(rel2,img2);
    var c4= new ComplexNumber(rel3,img3);
    var somme=c3.add(c4);

    document.getElementById("somme").innerHTML= somme.toString();

}

// on suppose les nombres complexe (rele+i imge)*(rele0 + i imge0)
function produitcomplexe(){
    var rel4=parseFloat(document.getElementById("rel4").value);
    var img4=parseFloat(document.getElementById("img4").value);
    var rel5=parseFloat(document.getElementById("rel5").value);
    var img5=parseFloat(document.getElementById("img5").value);
    var c5= new ComplexNumber(rel4,img4);
    var c6 = new ComplexNumber(rel5, img5);
    var produit= c5.times(c6);

    document.getElementById("produit").innerHTML= produit.toString();

}


// on suppose les nombres complexe (real+i imag)/(real0 + i imag0)

function quotientcomplexe(){
    var rel6=parseFloat(document.getElementById("rel6").value);
    var img6=parseFloat(document.getElementById("img6").value);
    var rel7=parseFloat(document.getElementById("rel7").value);
    var img7=parseFloat(document.getElementById("img7").value);
    var c7= new ComplexNumber(rel6,img6);
    var c8 = new ComplexNumber(rel7, img7);
    var quotient= c7.divided(c8);

    document.getElementById("quotient").innerHTML= quotient.toString();

}













function calculateRoots() {
    const rel8 = parseFloat(document.getElementById("rel8").value);
    const img8 = parseFloat(document.getElementById("img8").value);
    const rootDegree = parseInt(document.getElementById("rootDegree").value);

    const modulus = Math.sqrt(rel8 * rel8 + img8 * img8);
    const argument = Math.atan2(img8, rel8);
    
    const roots = [];

    for (let k = 0; k < rootDegree; k++) {
        const rootModulus = Math.pow(modulus, 1 / rootDegree);
        const rootArgument = (argument + 2 * Math.PI * k) / rootDegree;
        const rootReal = rootModulus * Math.cos(rootArgument);
        const rootImaginary = rootModulus * Math.sin(rootArgument);

        roots.push(formatComplexNumber(rootReal, rootImaginary));
    }

    document.getElementById("roots").textContent = roots.join("  ;    ");
}

function formatComplexNumber(real, imaginary) {
    if (imaginary >= 0) {
        return "(  "+  real + " + " + imaginary + "i   )     ";
    } else {
        return "(   "+ real + " - " + Math.abs(imaginary) + "i      ) ";
    }
}





