let net;
let model;
const scale = 10;

// adding or removing from this list will add/remove the number of values
// we get from mobilnet
const statNames = ['Beauty','Sourness','Stubble','Flakiness','Courage','Glow','literacy'];

const img1 = document.getElementById('img1');

async function generateStatsFromFile(file){
  // Make a prediction through the model on our image.
  const img1 = document.getElementById('img1');
  img1.style.display="block";
  img1.file = file;

  const reader = new FileReader();
  reader.onload = (function(i) { return function(e) { i.src = e.target.result; }; })(img1);
  reader.readAsDataURL(file);

  const r1 = model.predict(net.infer(img1, true));
  d1 = await r1.data();
  const p1 = document.getElementsByClassName('stats1')[0];
  // clear existing stats
  p1.innerHTML = '';
  for (let i = 0; i < d1.length; i++) p1.innerHTML = p1.innerHTML + ('<p>' + statNames[i] + ': ' + Math.round(d1[i]*scale) + '</p>');
}

//drag n drop
function dragOverHandler(ev) {ev.preventDefault();}
function dropHandler(ev) {
  ev.preventDefault();
  if (ev.dataTransfer.items) {
    // just grabbing the first item
    if (ev.dataTransfer.items[0].kind === 'file') {
      var file = ev.dataTransfer.items[0].getAsFile();
      if (file.type.startsWith('image/'))
        generateStatsFromFile(file);
    }
  } 
}

async function loadModel() {
  
  // just grabbed a random thing that said PCA could be a single layer network ,,
  // probably we should train some kind of autoencoder

  model = tf.sequential();
  // To simulate PCA we use 1 hidden layer with a linear (relu) activation
  const encoder = tf.layers.dense({
    units: statNames.length, 
    batchInputShape:[1,256], // match size of embeds that we get from mobilnet
    activation: 'relu',
    kernelInitializer:tf.initializers.randomNormal({'seed':2234}), // setting a seed gets consistent vals
    biasInitializer:"ones"});
  // const decoder = tf.layers.dense({units: 256, activation: 'relu'});

  model.add(encoder);
  // model.add(decoder); 
  await model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

  console.log('Loading mobilenet..');

  // Load the model.
  const version = 1;
  const alpha = 0.25; // less output embeds
  net = await mobilenet.load({version, alpha});
  console.log('Successfully loaded model');
}

loadModel();