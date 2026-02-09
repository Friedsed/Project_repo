var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");
var increment = 0;
var arrowLeft = false;
var arrowUp = false;
var arrowRight = false;
var arrowDown = false;
var spacebarPressed = false;
var elapsedTime;
const startingVelocity = 0;
var currVelocity = 0;
let carLength = 59.165;
let bufferLength = 7;
let carHeight = 25.165;

let isKeyed = true;

let maxVelocity =  330; // in km/h
let prevMaxvelo = maxVelocity;
maxVelocity = (maxVelocity/3.6)/(carLength) // convert to m/s divide by car length to get pixels per second
// console.log(maxVelocity)

let time = 7; // time to acclerate to max velocity

let acceleration = (maxVelocity/(time))*0.1; // vf = vi + at but vi = 0 so vf/t = a; // convert to m/0.1 s since we update the acceleration every 0.25s since if it was ever 1m/s it becomes too inaccurate
let msaccel = ((prevMaxvelo/3.6)/time);
let timeElapsed = 0;
let isAccelerating = false;
const gravity = 9.81;
var backgroundImage = new Image();
backgroundImage.src = "images/racingtrack.png";
var startTime = Date.now();

// prevents arrow down scrolling
window.addEventListener("keydown", function(e) {
  if(["Space","ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].indexOf(e.code) > -1) {
      e.preventDefault();
  }
}, false);
//  Update Values Below
setInterval(function() {
  document.getElementById("currSpeed").innerHTML = (Math.round((maxVelocity)*3.6*carLength*0.621371)) + " mph";
  document.getElementById("acceleration").innerHTML = (Math.round(((prevMaxvelo/3.6)/time))) + " m/s^2";
  document.getElementById("cofric").innerHTML = cofric;
}, 100); // run the function every 100 millisecond

window.onload = function() {
  WebFont.load({
    google: {
      families: ['Montserrat'],
      families: ['Noto Sans Math']
      
    },
    active: function() {
      // Font is now loaded and ready to use
    }
  });

  // Your canvas code here
};

backgroundImage.onload = function() {
  // Draw the image on the canvas
  ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
}
// creates a class that creates an element
class createElement {
  constructor(element, x, y, w, h) {
    this.element = element;
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
  }
}
// sets position of car
function setpositionelements(car) {
  var a = document.getElementById(car.element);
  a.style.left = car.x + 'px';
  a.style.top = car.y + 'px';
}

function countMilliseconds() {
  var startTime = new Date().getTime(); // get the current time in milliseconds
  setInterval(function() {
    elapsedTime = new Date().getTime() - startTime; // calculate the elapsed time
    
  }, 1); // run the function every 1 millisecond
}
countMilliseconds();
// get distance function between 2 objects, used for collision detection
function getDistance(x1, y1, x2, y2){

var xDistance = x2 - x1; 
var yDistance = y2 - y1; 

return Math.sqrt(Math.pow(xDistance, 2) + Math.pow(yDistance, 2));

}
// set boundaries to prevent car from going into water and off track
function Boundaries() {
  // var pixelDataHigh = ctx.getImageData(car.x+carLength, car.y+carHeight, 1, 1).data;
  // var pixelColorHigh = `rgb(${pixelDataHigh[0]}, ${pixelDataHigh[1]}, ${pixelDataHigh[2]})`;
  // var pixelDataMid = ctx.getImageData(car.x+(carLength/2), car.y+(carHeight/2), 1, 1).data;
  // var pixelColorMid = `rgb(${pixelDataMid[0]}, ${pixelDataMid[1]}, ${pixelDataMid[2]})`;
  // var pixelDataLow = ctx.getImageData(car.x, car.y, 1, 1).data;
  // var pixelColorLow = `rgb(${pixelDataLow[0]}, ${pixelDataLow[1]}, ${pixelDataLow[2]})`;
  // console.log(pixelColorHigh)
  // var tolerance = 50;
  // var isSimilar = isColorSimilar(pixelColorHigh, "rgb(255, 0, 0)", tolerance);
  // var isSimilar2 = isColorSimilar(pixelColorMid, "rgb(255, 0, 0)", tolerance);
  // var isSimilar3 = isColorSimilar(pixelColorLow, "rgb(255, 0, 0)", tolerance);
  // if (isSimilar || isSimilar2 || isSimilar3) {
  //   console.log("COLLISION DETECTED")
  //   car.x-=getDx();
  //   car.y-=getDy();
  // }

  let dx2 = getDx();
  let dy2 = getDy();
  if (isNaN(dx2)) {
    dx2 = currVelocity;
  }
  if (isNaN(dy2)) {
    dy2 = 0;
  }
  const centerX = 460;
  const centerY = 285;
  ctx.fillStyle = 'red';

// Draw a circle
ctx.beginPath();
ctx.arc(centerX, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();
  const innerRadius = 120;

  const centerSemi2X = 612
  ctx.beginPath();
ctx.arc(centerSemi2X, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();
const centerSemi1X = 270
ctx.beginPath();
ctx.arc(centerSemi1X, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();
const centerBTX = 365
ctx.beginPath();
ctx.arc(centerBTX, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();
const centerBTX2 = 545
ctx.beginPath();
ctx.arc(centerBTX2, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();
const centerOuter1 = 60;
ctx.beginPath();
ctx.arc(centerOuter1, centerY, 10, 0, 2 * Math.PI);
// ctx.fill();

  // console.log(getDistance(car.x, car.y,centerX,centerY))
  let currRotation = getRotation();
  let yDistance = centerY-car.y;
   if (getDistance(car.x, car.y,centerX,centerY) < innerRadius){
 
    const angle = Math.atan2(car.y - centerY, car.x - centerX);
const forceX = Math.cos(angle) * currVelocity;
    const forceY = Math.sin(angle) * currVelocity;
    if (arrowDown) {
      dx2-=forceX;
      dy2-=forceY;
    }else{
    dx2 += forceX;
    dy2 += forceY;
    }
    return {dx2,dy2};
   }

  if ( Math.sqrt(Math.pow(yDistance, 2)) >=160){
    const angle = Math.atan2(car.y - centerY, car.x - centerX);
    const forceX = Math.cos(angle) * currVelocity;
        const forceY = Math.sin(angle) * currVelocity;
        if (arrowDown) {
          dx2 += forceX;
          dy2 += forceY;
        }else{
        dx2 -= forceX;
        dy2 -= forceY;
        }
        return {dx2,dy2};
  }

  if (getDistance(car.x, car.y,centerSemi1X,centerY) >= 167 && car.x <300){
    const angle = Math.atan2(car.y - centerY, car.x - centerX);
    const forceX = Math.cos(angle) * currVelocity;
        const forceY = Math.sin(angle) * currVelocity;
        if (arrowDown) {
          dx2 += forceX;
          dy2 += forceY;
        }else{
        dx2 -= forceX;
        dy2 -= forceY;
        }
        return {dx2,dy2};
      }

      if (getDistance(car.x, car.y,centerSemi2X,centerY) >= 167 && car.x >=670 && car.x <830){
        const angle = Math.atan2(car.y - centerY, car.x - centerX);
        const forceX = Math.cos(angle) * currVelocity;
            const forceY = Math.sin(angle) * currVelocity;
            if (arrowDown) {
              dx2 += forceX;
              dy2 += forceY;
            }else{
            dx2 -= forceX;
            dy2 -= forceY;
            }
            return {dx2,dy2};
          }
// if (getDistance(car.x, car.y,centerOuter1,centerY) <60){
//   const angle = Math.atan2(car.y - centerY, car.x - centerOuter1);
//   const forceX = Math.cos(angle) * currVelocity;
//       const forceY = Math.sin(angle) * currVelocity;
//       if (arrowDown) {
//         dx2 -= forceX;
//         dy2 -= forceY;
//       }else{
//       dx2 += forceX;
//       dy2 += forceY;
//       }
//       return {dx2,dy2};
// }



   if (getDistance(car.x, car.y,centerSemi1X,centerY) < innerRadius ){
    const angle = Math.atan2(car.y - centerY, car.x - centerSemi1X);
const forceX = Math.cos(angle) * currVelocity;
    const forceY = Math.sin(angle) * currVelocity;
    if (arrowDown) {
      dx2 -= forceX;
      dy2 -= forceY;
    }else{
    dx2 += forceX;
    dy2 += forceY;
    }
    
    return {dx2,dy2};
   }

   if (getDistance(car.x, car.y,centerSemi2X,centerY) < innerRadius ){
    const angle = Math.atan2(car.y - centerY, car.x - centerSemi2X);
const forceX = Math.cos(angle) * currVelocity;
    const forceY = Math.sin(angle) * currVelocity;
    if (arrowDown) {
      dx2-=forceX;
      dy2-=forceY;
    }else{
    dx2 += forceX;
    dy2 += forceY;
    }
    
    return {dx2,dy2};
   }

   if (getDistance(car.x, car.y,centerBTX,centerY) < innerRadius ){
    const angle = Math.atan2(car.y - centerY, car.x - centerBTX);
const forceX = Math.cos(angle) * currVelocity;
    const forceY = Math.sin(angle) * currVelocity;
    if (arrowDown) {
      dx2-=forceX;
      dy2-=forceY;
    }else{
    dx2 += forceX;
    dy2 += forceY;
    }
    
    return {dx2,dy2};
   }



   if (getDistance(car.x, car.y,centerBTX2,centerY) < innerRadius ){
    const angle = Math.atan2(car.y - centerY, car.x - centerBTX2);
const forceX = Math.cos(angle) * currVelocity;
    const forceY = Math.sin(angle) * currVelocity;
    if (arrowDown) {
      dx2-=forceX;
      dy2-=forceY;
    }else{
    dx2 += forceX;
    dy2 += forceY;
    }
    
    return {dx2,dy2};
   }
   return null;

  // console.log(getDistance(car.x, car.y,centerOvalX,centerOvalY))
}
// check if colours are the same used for collision detect
function isColorSimilar(color1, color2, tolerance) {
  var rgb1 = color1.match(/\d+/g);
  var rgb2 = color2.match(/\d+/g);
  for (var i = 0; i < 3; i++) {
    if (Math.abs(rgb1[i] - rgb2[i]) > tolerance) {
      return false;
    }
  }
  return true;
}
// show the car element
function showelements() {
  setpositionelements(car);
}
// get distance between two points
function getDistance(x1, y1, x2, y2) {
  var xDistance = x2 - x1;
  var yDistance = y2 - y1;
  return Math.sqrt(Math.pow(xDistance, 2) + Math.pow(yDistance, 2));
}
// arrow key event handlers used for control of car
document.addEventListener("keydown", keyDownHandler, false);
document.addEventListener("keyup", keyUpHandler, false);
function keyDownHandler(e) {
  if ((e.key == "ArrowRight" || e.key == "ArrowRight")) {
    arrowRight = true;
  }
  else if (e.key == "ArrowLeft" || e.key == "ArrowLeft" ) {
    arrowLeft = true;
  } else if (e.key == "ArrowUp" || e.key == "ArrowUp") {
    arrowUp = true;
  }
  else if ((e.key == "ArrowDown" || e.key == "ArrowDown")) {
    // arrowDown = true;
  }
  else if (e.key == " " || e.key == "Space") {
    spacebarPressed = true;
  }
}
function keyUpHandler(e) {
  if (e.key == "ArrowRight" || e.key == "ArrowRight") {
    arrowRight = false;
  }
  else if (e.key == "ArrowLeft" || e.key == "ArrowLeft") {
    arrowLeft = false;
  }
  else if (e.key == "ArrowUp" || e.key == "ArrowUp") {
    arrowUp = false;
  }
  else if (e.key == "ArrowDown" || e.key == "ArrowDown") {
    arrowDown = false;
  }
  else if (e.key == " " || e.key == "Space") {
    spacebarPressed = false;
  }
}
// get dx and dy for rotation of car
let rotationincrementleft = 0;
let rotationincrementright = 0;
function getDx() {
  var angle =   getRotation() * Math.PI / 180;
  var dx = Math.cos(angle) * currVelocity;
  return dx;
}
function getDy(){
  var angle =   getRotation() * Math.PI / 180;
  var dy = Math.sin(angle) * currVelocity;
return dy;
}
// set top speed from input boxes
function setTopSpeed(){
  isEnter2 = false;
  isKeyed = true;
 let hp = parseFloat(document.getElementById("hp").value);
 let w = parseFloat(document.getElementById("w").value);
 let h = parseFloat(document.getElementById("h").value);
 let dragc = parseFloat(document.getElementById("dragc").value);
console.log(hp)
 if (isNaN(hp)|| isNaN(w) || isNaN(h) || isNaN(dragc)){
  alert("Please fill all the fields");
  return;
 }
 let p = 1.225;

 if (document.getElementById("accel").value != ""){
  time = document.getElementById("accel").value
 }
 hp = hp*745.7;
// console.log(w + " " + h + " " + dragc + " " + p)
//  console.log(w*h*dragc*0.5*p)
//  console.log((hp)/(w*h*dragc*0.5*p))
maxVelocity = Math. cbrt((hp)/(w*h*dragc*0.5*p))*3.6;
prevMaxvelo = maxVelocity;
maxVelocity = (maxVelocity/3.6)/(carLength)
 acceleration = (maxVelocity/(time))*0.1;
 msaccel = ((prevMaxvelo/3.6)/time);
 console.log("prev max"+prevMaxvelo)
 console.log("New max "+maxVelocity)

FN = mass*gravity;

}
// modify velocity with acceleration
function modifyVelo(){
  // console.log(elapsedTime)
  currVelocity += acceleration;
  if (currVelocity > maxVelocity) {
    currVelocity = maxVelocity;
  }

  if (currVelocity < 0){
    currVelocity = 0;
    isAccelerating = false;
  }

 
}

// slow down the velocity if brakind force applied
function slowVelo(){
  currVelocity -= acceleration;
  if (currVelocity <= 0) {
    currVelocity = 0;
  }
}
let intervalId;
let intervalId2;




// get rotation of car
function getRotation(){
  var el = document.getElementById("car");
var st = window.getComputedStyle(el, null);
var tr = st.getPropertyValue("-webkit-transform") ||
         st.getPropertyValue("-moz-transform") ||
         st.getPropertyValue("-ms-transform") ||
         st.getPropertyValue("-o-transform") ||
         st.getPropertyValue("transform") ||
         "FAIL";

// With rotate(30deg)...
// matrix(0.866025, 0.5, -0.5, 0.866025, 0px, 0px)

// rotation matrix - http://en.wikipedia.org/wiki/Rotation_matrix
if (tr !== 'none') {
var values = tr.split('(')[1].split(')')[0].split(',');
var a = values[0];
var b = values[1];
var c = values[2];
var d = values[3];

var scale = Math.sqrt(a*a + b*b);


// arc sin, convert from radians to degrees, round
var sin = b/scale;
// next line works for 30deg but not 130deg (returns 50);
// var angle = Math.round(Math.asin(sin) * (180/Math.PI));
var angle = Math.round(Math.atan2(b, a) * (180/Math.PI));

}
return angle;
}
var carx = 345;
var cary = 425;
var car = new createElement('car', carx, cary, 100, 100);
// run the simulation
function draw() {
  controls();
  showelements();
}
setInterval(draw, 1); 

// EVERYTHING BELOW IS FOR THE SPEDOMETER (RPM IS NOT ACCURATE JUST FOR VISUAL)
let spedometerCanvas = document.getElementById("spedometer");
let ctx2 = spedometerCanvas.getContext("2d");

var speedGradient = ctx2.createLinearGradient(0, 500, 0, 0);
speedGradient.addColorStop(0, '#00b8fe');
speedGradient.addColorStop(1, '#41dcf4');

var rpmGradient = ctx2.createLinearGradient(0, 500, 0, 0);
rpmGradient.addColorStop(0, '#f7b733');
rpmGradient.addColorStop(1, '#fc4a1a');
//rpmGradient.addColorStop(1, '#EF4836');

function speedNeedle(rotation) {
    ctx2.lineWidth = 2;

    ctx2.save();
    ctx2.translate(250, 250);
    ctx2.rotate(rotation);
    ctx2.strokeRect(-130 / 2 + 170, -1 / 2, 135, 1);
    ctx2.restore();

    rotation += Math.PI / 180;
}

function rpmNeedle(rotation) {
    ctx2.lineWidth = 2;

    ctx2.save();
    ctx2.translate(250, 250);
    ctx2.rotate(rotation);
    ctx2.strokeRect(-130 / 2 + 170, -1 / 2, 135, 1);
    ctx2.restore();

    rotation += Math.PI / 180;
}

function drawMiniNeedle(rotation, width, speed) {
    ctx2.lineWidth = width;

    ctx2.save();
    ctx2.translate(250, 250);
    ctx2.rotate(rotation);
    ctx2.strokeStyle = "#333";
    ctx2.fillStyle = "#333";
    ctx2.strokeRect(-20 / 2 + 220, -1 / 2, 20, 1);
    ctx2.restore();

    let x = (250 + 180 * Math.cos(rotation));
    let y = (250 + 180 * Math.sin(rotation));

    ctx2.font = "200 20px Montserrat";
    ctx2.fillText(speed, x, y);

    rotation += Math.PI / 180;
}

function calculateSpeedAngle(x, a, b) {
    let degree = (a - b) * (x) + b;
    let radian = (degree * Math.PI) / 180;
    return radian <= 1.45 ? radian : 1.45;
}

function calculateRPMAngel(x, a, b) {
    let degree = (a - b) * (x) + b;
    let radian = (degree * Math.PI) / 180;
    return radian >= -0.46153862656807704 ? radian : -0.46153862656807704;
}

function drawSpeedo(speed, gear, rpm, topSpeed) {

    ctx2.clearRect(0, 0, 500, 500);

    ctx2.beginPath();
    ctx2.fillStyle = 'rgba(0, 0, 0, .9)';
    ctx2.arc(250, 250, 240, 0, 2 * Math.PI);
    ctx2.fill();
    ctx2.save()
    ctx2.restore();
    ctx2.fillStyle = "#FFF";
    ctx2.stroke();

    ctx2.beginPath();
    ctx2.strokeStyle = "#333";
    ctx2.lineWidth = 10;
    ctx2.arc(250, 250, 100, 0, 2 * Math.PI);
    ctx2.stroke();

    ctx2.beginPath();
    ctx2.lineWidth = 1;
    ctx2.arc(250, 250, 240, 0, 2 * Math.PI);
    ctx2.stroke();

    ctx2.font = "200 70px Montserrat";
    ctx2.textAlign = "center";
    ctx2.fillText(speed, 250, 220);

    ctx2.font = "200 15px Montserrat";
    ctx2.fillText("mph", 250, 235);

    if (gear == 0 && speed > 0) {
        ctx2.fillStyle = "#999";
        ctx2.font = "200 70px Montserrat";
        ctx2.fillText('R', 250, 460);

        ctx2.fillStyle = "#333";
        ctx2.font = "50px Montserrat";
        ctx2.fillText('N', 290, 460);
    } else if (gear == 0 && speed == 0) {
        ctx2.fillStyle = "#999";
        ctx2.font = "200 70px Montserrat";
        ctx2.fillText('N', 250, 460);

        ctx2.fillStyle = "#333";
        ctx2.font = "200 50px Montserrat";
        ctx2.fillText('R', 210, 460);

        ctx2.font = "200 50px Montserrat";
        ctx2.fillText(parseInt(gear) + 1, 290, 460);
    } else if (gear - 1 <= 0) {
        ctx2.fillStyle = "#999";
        ctx2.font = "200 70px Montserrat";
        ctx2.fillText(gear, 250, 460);

        ctx2.fillStyle = "#333";
        ctx2.font = "50px Montserrat";
        ctx2.fillText('R', 210, 460);

        ctx2.font = "200 50px Montserrat";
        ctx2.fillText(parseInt(gear) + 1, 290, 460);
    } else {
        ctx2.font = "200 70px Montserrat";
        ctx2.fillStyle = "#999";
        ctx2.fillText(gear, 250, 460);

        ctx2.font = "200 50px Montserrat";
        ctx2.fillStyle = "#333";
        ctx2.fillText(gear - 1, 210, 460);
        if (parseInt(gear) + 1 < 7) {
            ctx2.font = "200 50px Montserrat";
            ctx2.fillText(parseInt(gear) + 1, 290, 460);
        }
    }

    ctx2.fillStyle = "#FFF";
    for (var i = 10; i <= Math.ceil(topSpeed / 20) * 20; i += 10) {
        // console.log();
        drawMiniNeedle(calculateSpeedAngle(i / topSpeed, 83.07888, 34.3775) * Math.PI, i % 20 == 0 ? 3 : 1, i%20 == 0 ? i : '');
        
        if(i<=100) { 
            drawMiniNeedle(calculateSpeedAngle(i / 47, 0, 22.9183) * Math.PI, i % 20 == 0 ? 3 : 1, i % 20 ==
            0 ?
            i / 10 : '');
        }
    }

    ctx2.beginPath();
    ctx2.strokeStyle = "#41dcf4";
    ctx2.lineWidth = 25;
    ctx2.shadowBlur = 20;
    ctx2.shadowColor = "#00c6ff";

    ctx2.strokeStyle = speedGradient;
    ctx2.arc(250, 250, 228, .6 * Math.PI, calculateSpeedAngle(speed / topSpeed, 83.07888, 34.3775) * Math.PI);
    ctx2.stroke();
    ctx2.beginPath();
    ctx2.lineWidth = 25;
    ctx2.strokeStyle = rpmGradient;
    ctx2.shadowBlur = 20;
    ctx2.shadowColor = "#f7b733";

    ctx2.arc(250, 250, 228, .4 * Math.PI, calculateRPMAngel(rpm / 4.7, 0, 22.9183) * Math.PI, true);
    ctx2.stroke();
    ctx2.shadowBlur = 0;


    ctx2.strokeStyle = '#41dcf4';
    speedNeedle(calculateSpeedAngle(speed / topSpeed, 83.07888, 34.3775) * Math.PI);

    ctx2.strokeStyle = rpmGradient;
    rpmNeedle(calculateRPMAngel(rpm / 4.7, 0, 22.9183) * Math.PI);

    ctx2.strokeStyle = "#000";
}

// set speed of spedometer
function setSpeed () {

let speedM = 0;
let gear = 0;
let rpm = 0
setInterval(function(){
// if (speedM > 160){
// speedM = 0;
// rpm = 0;
// }
let speed =  Math.round((currVelocity*3.6*carLength)*0.621371)
if (speed == 0){
  gear = 0
}
if (speed > 1 && speed< 30){
gear = 1;
rpm += .03;
} else if (speed > 30 && speed < 50) {
gear = 2;
rpm += .03;
  } else if (speed > 50 &&   speed < 70) {
gear = 3;
rpm += .03;
} else if (speed > 70 &&   speed < 100)      {
gear = 4;
rpm += .03;
  } else if (speed > 200)      {
gear = 5;
rpm += .03;
}

if (rpm < 1){
rpm += .03; 
}
drawSpeedo(speed,gear,rpm,200);

}, 40);

}

document.addEventListener('DOMContentLoaded', function() {
  ctx2.scale(0.65,0.65);
//setInterval(setSpeed, 2000)
//renderCanvas();
setSpeed();
//drawSpeedo(120,4,.8,160);
}, false);
  // DYNAMICS
  let mass = 780;
  // console.log("Mass is " + mass + " kg")
  let ma = 0
  let cofric = 1.7 // standard coefficient of friction for f1 tires
  let FN = mass*gravity;
  let staticFric = 0;
  let forceRizzistance = 0;
  let brakingForce = 20000;
let sideview = document.getElementById('sideview');
let sidectx = sideview.getContext('2d');
let sideImg = new Image();
sideImg.src = "images/beachbg2.png";

let carnonA = new Image();
carnonA.src = "images/f1.gif";
carnonA.onload = function() {
  sidectx.drawImage(carnonA, 170, 220, 150, 150);
};


// use a gif for the 2nd canvas

var width = window.innerWidth;
var height = window.innerHeight;

var stage = new Konva.Stage({
  container: 'sideview',
  width: width,
  height: height,
});

var layer = new Konva.Layer();
stage.add(layer);
setInterval(function() {
  if (currVelocity > 0){
function onDrawFrame(ctx, frame) {
  // update canvas size
  frame.delay = 100000000000000000000000;
  
      if (currVelocity >0){
        frame.delay = 10;
      }
  // update canvas that we are using for Konva.Image
  sidectx.drawImage(frame.buffer, 170, 220,150,150);
  drawForceArrows();

      
  // redraw the layer
  layer.draw();
}

 var gif = gifler('images/f1.gif').frames(sideview, onDrawFrame);
}
}, 500)
// draw resulted canvas into the stage as Konva.Image
var image = new Konva.Image({
  image: canvas,

});


layer.add(image);


  sidectx.drawImage(sideImg, 0, 0,sideview.width, sideview.height);
// calculate the various dynamics and kinematics forces

function calcMA(){
  if (!isAccelerating || currVelocity == maxVelocity){
    return 0;
  }
  console.log("prevMaxvelo is " + prevMaxvelo)
return ((prevMaxvelo/3.6)/time)*mass;
}

function calcFric(){
  return FN*cofric;
}

function calcRizzistance(){
  return staticFric-ma
}

let isEnter2 = false;
var errorElement = document.getElementById('error');
setInterval(function() {
 
  ma = calcMA();
 staticFric = calcFric();
  forceRizzistance = calcRizzistance();
  // check if the force of static friction is greater than the force required for acceleration if not then throw an error
 if (Math.abs(forceRizzistance) > staticFric){
isKeyed = false;
maxVelocity = 0;
var existingContent = errorElement.innerHTML.trim();

if (existingContent === '' && !isEnter2) {
  errorElement.innerHTML += '<article class="message is-danger">' +
    '<div class="message-header">' +
    '<p>Error</p>' +
    '<button class="delete" aria-label="delete"></button>' +
    '</div>' +
    '<div class="message-body">' +
    'Please enter a greater value for time to get to max speed, which results in a slower acceleration. If the car has a slower acceleration, it means that the force required to accelerate the car is lower. Consequently, the difference between the force of static friction and the force required for acceleration (F_resistance_total) would be smaller. Therefore, in this simplified model, a slower acceleration would correspond to a lower resistance force acting on the car.' +
    '</div>' +
    '</article>';
  
    }
setTimeout(function() {
  errorElement.innerHTML = '';
  isEnter2 = true;
}, 10000); // 10 seconds = 10000 milliseconds


  
}


//   while (staticFric < forceRizzistance){
//     mass+=20;
//      FN = mass*gravity;
//      staticFric = calcFric();
//   forceRizzistance = calcRizzistance();
// console.log(staticFric + "IN LOOP " + forceRizzistance)
// if (staticFric > forceRizzistance){
//   console.log("in loop ur stupid")
//   return;
// }
//   }

  // console.log("MA is " +ma)
  // console.log("FFS is "+staticFric)
  // console.log("FR is " +forceRizzistance)
  drawForceArrows();

}, 100);

// calculate for braking
let ma2 = staticFric-forceRizzistance-brakingForce;
let acceleration2 = ((ma2/mass)/3.6)*0.1; // convert acceleration back
console.log(ma2)

console.log(acceleration2)
let isEntered = false;
function controls() {
  ma2 = staticFric-Math.abs(forceRizzistance)-brakingForce;
 acceleration2 = ((ma2/mass)/3.6)*0.01; 
 if (brakingForce+ Math.abs(forceRizzistance) < staticFric ){
  isKeyed = false;
  // maxVelocity = 0;
currVelocity = 0;
  var existingContent = errorElement.innerHTML.trim();
  
  if (existingContent === '' && !isEntered) {
    errorElement.innerHTML += '<article class="message is-danger">' +
      '<div class="message-header">' +
      '<p>Error</p>' +
      '<button class="delete" aria-label="delete"></button>' +
      '</div>' +
      '<div class="message-body">' +
      'Please enter a greater value for the magnitude of the braking force so that it can decelerate the car' +
      '</div>' +
      '</article>';
    isEntered = true;
      }
  setTimeout(function() {
    errorElement.innerHTML = '';
  }, 10000); // 10 seconds = 10000 milliseconds
  
  
    arrowUp = false;
    arrowLeft = false;
    arrowRight = false;
  }
  
  //  console.log(currVelocity)
  let dx = 0;
  let dy = 0; 
    let carele = document.getElementById("car");
  // add if statement
  if (Boundaries() != null){
    let {dx2, dy2} = Boundaries();
   dx = dx2
   dy = dy2
  }else{
     dx = getDx();
     dy = getDy();
  }
   if (isNaN(dx)) {
      dx = currVelocity;
    }
    if (isNaN(dy)) {
      dy = 0;
    }
    if (arrowUp && isKeyed) {
          acceleration =  (maxVelocity/(time))*0.1;

          car.x += dx;
      car.y += dy;
      isAccelerating = true;
      clearInterval(intervalId2);
      intervalId2 = null;
      if (!intervalId){
        intervalId =setInterval(modifyVelo, 100);
      }
    }
   
    if (arrowLeft && isKeyed) {
          isAccelerating = true;
      // modifyVelo();
      carele.style.transform = `rotate(${rotationincrementleft}deg)`;
      rotationincrementleft--;
    }
    if (arrowRight && isKeyed) {
          isAccelerating = true;
      // modifyVelo();
      carele.style.transform = `rotate(${rotationincrementleft}deg)`;
      rotationincrementleft++;
    }
    if (!(arrowUp) && currVelocity > 0 && isKeyed) {
      acceleration = acceleration2
      car.x += dx;
      car.y += dy;
      isAccelerating = true;
      clearInterval(intervalId2);
      intervalId2 = null;
      if (!intervalId){
        intervalId =setInterval(modifyVelo, 100);
      }

  }
  
  
  }

  // draw arrows for forces
  function drawForceArrows() {
    if (forceRizzistance >= 0) {
      forceRizzistance = -forceRizzistance;
    }
    
    
  // add the text for the forces on the 2nd canvas
    
    let text3 = document.getElementById("fg");
    let fgval = document.getElementById("fgval");
    fgval.innerHTML = (Math.abs(Math.round(FN)) + "N").italics();
    
    let text4 = document.getElementById("fn");
    let fnval = document.getElementById("fnval");
    fnval.innerHTML = (Math.abs(Math.round(FN)) + "N").italics();
    
    var canvasRect = sideview.getBoundingClientRect();
    var x = 370;
    let x2 = 100;
    let y2 = 250;
    let x3 = 225;
    let y3 = 360;
    let x4 = 225;
    let y4 = 150;
    var y = 250;
    
    let fbval = document.getElementById("fbval");
    fbval.style.display = "none";
    
   
    text3.style.position = "fixed";
    text3.style.left = (canvasRect.left + x3) + "px";
    text3.style.top = (canvasRect.top + y3) + "px";
    
    text4.style.position = "fixed";
    text4.style.left = (canvasRect.left + x4) + "px";
    text4.style.top = (canvasRect.top + y4) + "px";
    
    fgval.style.position = "fixed";
    fgval.style.left = (canvasRect.left + x3) + "px";
    fgval.style.top = (canvasRect.top + y3 + 20) + "px";
    
    fnval.style.position = "fixed";
    fnval.style.left = (canvasRect.left + x4) + "px";
    fnval.style.top = (canvasRect.top + y4 + 20) + "px";
    
    let fb = document.getElementById("fb");
    fb.style.left = (canvasRect.left + x2) + "px";
    fb.style.top = (canvasRect.top + y2 + 65) + "px";
    fb.style.display = "none";
    var text = document.getElementById("ffs");
    let ffsval = document.getElementById("ffsval");
    
    let text2 = document.getElementById("fr");
    let ffrval = document.getElementById("ffrval");
    if (isAccelerating){
      text.style.display="";
      ffsval.style.display="";
      text2.style.display="";
      ffrval.style.display="";
      ffsval.innerHTML = (Math.round(staticFric) + "N").italics();
      
      ffrval.innerHTML = (Math.abs(Math.round(forceRizzistance)) + "N").italics();
      text.style.left = (canvasRect.left + x) + "px";
      text.style.top = (canvasRect.top + y) + "px";
      
      ffsval.style.left = (canvasRect.left + x) + "px";
      ffsval.style.top = (canvasRect.top + y + 20) + "px";
      
      text2.style.left = (canvasRect.left + x2) + "px";
      text2.style.top = (canvasRect.top + y2) + "px";
      
      ffrval.style.left = (canvasRect.left + x2) + "px";
      ffrval.style.top = (canvasRect.top + 20 + y2) + "px";
      
      drawHorizontalArrow(sidectx, 240, 300, forceRizzistance/100, 10);
      drawHorizontalArrow(sidectx, 240, 300, staticFric/100, 10);
    }else{
      text.style.display="none";
      ffsval.style.display="none";
      text2.style.display="none";
      ffrval.style.display="none";
    }
    drawVerticalArrow(sidectx, 240, 300, FN/150, 10);
    drawVerticalArrowUp(sidectx, 240, 300, FN/150, 10);
    
    if (!(arrowUp) && currVelocity > 0) {
      fb.style.display = "";
      fbval.style.display = "";
      
      drawHorizontalArrow(sidectx, 240, 310, -(brakingForce/100), 10);
      
      fbval.innerHTML = (Math.abs(Math.round(brakingForce)) + "N").italics();
      fbval.style.left = (canvasRect.left + x2) + "px";
      fbval.style.top = (canvasRect.top + y2 + 85) + "px";
    }
  }
  
  // draw arrows for vectors on 2nd canvas
function drawVerticalArrowUp(canvas, x, y, height, arrowWidth) {
  var ctx = canvas;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y - height);
  ctx.lineTo(x - arrowWidth, y - height + arrowWidth);
  ctx.moveTo(x, y - height);
  ctx.lineTo(x + arrowWidth, y - height + arrowWidth);
  ctx.stroke();
}





function drawVerticalArrow(canvas, x, y, height, arrowWidth) {
  var ctx = canvas;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x - arrowWidth, y + height - arrowWidth);
  ctx.moveTo(x, y + height);
  ctx.lineTo(x + arrowWidth, y + height - arrowWidth);
  ctx.stroke();
}

function drawHorizontalArrow(ctx, startX, startY, length, arrowSize) {
  // Draw the line
 ctx.strokeStyle = "red";
 ctx.fillStyle = "red";
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(startX + length, startY);
  ctx.stroke();

  // Draw the arrowhead
  const endX = startX + length;
  const endY = startY;
  const angle = Math.atan2(0, length);
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX - arrowSize * Math.cos(angle - Math.PI / 6), endY - arrowSize * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(endX - arrowSize * Math.cos(angle + Math.PI / 6), endY - arrowSize * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}
// get braking force
function getBrakingForce(){
 let  brakingForce2 = parseInt(document.getElementById("brakingForce").value);
  if (isNaN(brakingForce2)) {
  alert("Please enter a number");
  return;
  }
  brakingForce = brakingForce2
   isEntered = false;
   isKeyed = true;

}
// set braking force
function setBrakingForce(){
  ma2 = staticFric-Math.abs(forceRizzistance)-brakingForce;
 acceleration2 = ((ma2/mass)/3.6)*0.01; 
 
  
 }

//  CHART JS BELOW
 var vVals = [];
 var ctx3 = document.getElementById("myChart").getContext("2d");
 // Initialize the line chart
 var chart = new Chart(ctx3, {
   type: 'line',
   data: {
     labels: [],
     datasets: [{
       label: 'Velocity',
       data: [],
       borderColor: 'blue',
       fill: false
     }]
   },
   options: {
     responsive: true,
     animation: false,
     scales: {
       x: {
         display: true,
         title: {
           display: true,
           text: 'Time (s)'
         }
       },
       y: {
         display: true,
         title: {
           display: true,
           text: 'Velocity (mph)'
         }
       }
     }
   }
 });

 function getVeloVals() {
   vVals.push(currVelocity*3.6*carLength*0.621371);

   // Update the chart data
   chart.data.labels.push(vVals.length); // Assuming time starts at 1 second and increments by 1 second
   chart.data.datasets[0].data.push(currVelocity*3.6*carLength*0.621371);

   // Update the chart
   chart.update();
 }

 // Example usage
 setInterval(function() {
   getVeloVals();
 }, 1000); // Update the chart every second with a random velocity value
 

 