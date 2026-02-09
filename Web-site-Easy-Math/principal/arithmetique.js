class Point2{
    constructor(x,y){
        this.x=x;
        this.y=y;
    }
    

    toString2(){
        return '('+this.x +' , ' +this.y + ')'
    }
}

// supposons a=bq+r divisions euclidienne de a par b 
// j'ai commencé a prendre les nombres r de 0 à a comme reste, pour chaque r je verifie si a-r est bdivisible par b.
// si oui je calcule le quotient q
// ensuite je coupe le if puis je sort pour calculer de reste


function diveuclid2(a,b){
    for (var i=0; i<a;i++){
        if ((a-i)%b===0){
            var a2=(a-i)/b;
            break;
    }
}
    a1= a-b*a2;
    var a3=new Point2(a1,a2);
    return a3;
}


// supposons l'equation ax+by=1
// j'ai utilisé l    notion de classe en arithmetique 1: supposé que je travail dans la classe de Z/bZ  
// 2 : j'ai trouve l'inverse de a dans Z/BZ qui est x=a1 en cherchant quelle nombre (i) inferieur à < b et a*i est congrue a 1 mod(b)  c'est ca le code dans le if
//   3: j'ai calcules y en utilisant l'equation y=(1-a*x)/b
function equdio(a,b){
    for(var i=0; i<Math.abs(b);i++)
    if((a*i)%b===1){
        var a1=i;
        break;
    }
    var a2=(-a*a1+1)/b;
    var a3=new Point2(a1,a2);
    return a3;
    
}




function pgcdRec(a,b){
    if(b==0){
          return a;
           }
          return pgcdRec(b,a%b);
      }

      function ppcmRec(a,b){
        return a*b/pgcdRec(a,b);
    }





function npremier_FA(n, i ) {
    if (n <= 2) {
        return (n == 2) ? true : false;
    }
    if (n % i == 0) {
        return false;
    }
    if (i * i > n) {
        return true;
    }
    return npremier_FA(n, i + 1);
}

function npremier(r){
    return npremier_FA(r,2);
}

function fact(a){
    var n=1;
    for (var i=1; i<=a ;i++ ){
        n=n*i;
    }return n;
}

function arrg(a,p){
    return fact(a)/fact(a-p);
}

function comb(a,p){
    return arrg(a,p)/fact(p);
}






// fi,n des fonctions initiales

function find_congruence(){
    var a00=parseInt(document.getElementById("a00").value);
    var b00=parseInt(document.getElementById("b00").value);

   
    var c00=a00%b00;
    

    document.getElementById("c00").innerHTML=a00+" est congrue à "+c00+" modulo "+ b00;

}

function calculate_gcd_lcm(){
    var a=parseInt(document.getElementById("a").value);
    var b=parseInt(document.getElementById("b").value);
   
    var d=pgcdRec(a,b);
    var e=ppcmRec(a,b);

    document.getElementById("d").innerHTML=d;
    document.getElementById("e").innerHTML=e;

}


function verify_prime(){
    var r=parseInt(document.getElementById("r").value);

    if(npremier(r)){
        var result=' est premier ';
    }
    else{
        var result=' n \' est pas premier';
    }

    document.getElementById("result").innerHTML=result;

}


function find_divisors(){
    var n0=parseInt(document.getElementById("n0").value);

    var divisors=[];

    for (var i=1; i<=n0;i++){
        if (n0%i===0){
            divisors.push(i);
            }
        }

        document.getElementById("answer").innerHTML= "Divisors:"+divisors.join(",");


}


function commundivisors(){
    var a1=parseInt(document.getElementById("a1").value);
    var a2=parseInt(document.getElementById("a2").value);
    
    var divisors1=[];

    // if(a1>a2){
    //     var o=a2;
    //     a2=a1;
    //     a1=o;
    // }
    for (var i=1; i<= Math.min(a1,a2) ; i++){
        if (a1%i===0 && a2%i===0){
            divisors1.push(i);
        }


    }
    document.getElementById("answer1").innerHTML="Commun divisors :"+divisors1.join(",");

}



function find_factorial(){
    var n=parseInt(document.getElementById("n").value);
    var fac=fact(n);
    document.getElementById("fac").innerHTML=fac;

}
// arrangement c'est permutation
function find_permutation(){
    var n1=parseInt(document.getElementById("n1").value);
    var p=parseInt(document.getElementById("p").value);
    var arrag=arrg(n1,p);
    document.getElementById("arrag").innerHTML=arrag;

}

function find_combination(){
    var n2=parseInt(document.getElementById("n2").value);
    var p1=parseInt(document.getElementById("p1").value);
    var combi=comb(n2,p1);
    document.getElementById("combi").innerHTML=combi;

}


function euclidean_division(){
    var a3=parseInt(document.getElementById("a3").value);
    var a4=parseInt(document.getElementById("a4").value);

        a5=diveuclid2(a3,a4);
        a6=a5.x;
        a7=a5.y;
        document.getElementById("a6").innerHTML=a6;
        document.getElementById("a7").innerHTML=a7;
        document.getElementById("a5").innerHTML= a5.toString2();
    }


    function find_diophantine_equation(){
        var a9=parseInt(document.getElementById("a9").value);
        var a10=parseInt(document.getElementById("a10").value);
    
            a11=equdio(a9,a10);
            a12=a11.x;
            a13=a11.y;
            document.getElementById("a12").innerHTML=a12+' + ('+a10+') k  ,   k c Z';
            document.getElementById("a13").innerHTML=a13+' -('+ a9+') k  ,   k c Z';
           
        }



        function find_diophantine_equation_2(){
            var a15=parseInt(document.getElementById("a15").value);
            var a16=parseInt(document.getElementById("a16").value);
            var a17=parseInt(document.getElementById("a17").value);
                if (a17%pgcdRec(a15,a16)===0){
                    var a21=a15/pgcdRec(a15,a16);
                    var a22=a16/pgcdRec(a15,a16);
                }
                a18=equdio(a21,a22);

                a19=(a18.x * a17)/pgcdRec(a15,a16);
                a20=(a18.y * a17)/pgcdRec(a15,a16);
                
                
                document.getElementById("a19").innerHTML=a19+' + ('+a22+') k  ,   k c Z';
                document.getElementById("a20").innerHTML=a20+' -('+a21+') k  ,   k c Z';
               
            }




            // Function to convert a number to the specified base
            function convertToBase() {

                
                
                // Prompt the user for input
                var a22 = parseInt(document.getElementById("a22").value);// le nombre
                var a23 = parseInt(document.getElementById("a23").value);// la base
                var a24=a22;
                var tab=[];
                do{
                    if(a22%a23<10){
                        tab.push(a22%a23);
                    }
                    else if(a22%a23===10){
                        tab.push('A');
                    }
                    else if(a22%a23===11){
                        tab.push('B');
                    }
                    else if(a22%a23===12){
                        tab.push('C');
                    }
                    else if(a22%a23===13){
                        tab.push('D');
                    }
                    else if(a22%a23===14){
                        tab.push('E');
                    }
                    else if(a22%a23===15){
                        tab.push('F');
                    }
                    else if(a22%a23===16){
                        tab.push('G');
                    }
                    else if(a22%a23===17){
                        tab.push('H');
                    }
                    else if(a22%a23===18){
                        tab.push('I');
                    }
                    
                    a22=parseInt(a22/a23);

                  
                }while( a22!== 0)

                tab=tab.reverse();
                document.getElementById("tab").innerHTML=a24+" en base "+a23+" est : "+tab.join("");           
                
            }

