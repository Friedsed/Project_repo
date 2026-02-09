


function calculate(){


    var func=math.parse(document.getElementById("func").value);
    var derivative=math.derivative(func).toString();
    var primitive=math.integral(func).toString();
    var result ="Derivative : " + derivative  +  " <br> Primitive " + primitive;
    document.getElementById("result").innerHTML = result;


}