// * Http for fetch with: python -m http.server 9090

const celsius = document.querySelector('#celsius');

const changeCelsiusToFahrenheit = () => {
  const celsiusValue = celsius.value;
  document.querySelector('.lbl-celsius').innerHTML = celsiusValue;

  if (model) {
    const tensor = tf.tensor1d([Number(celsiusValue)]);

    // Wait to get the prediction
    let prediction = model.predict(tensor).dataSync();
    prediction = Math.round(prediction);

    document.querySelector('.result').innerHTML = `${celsiusValue} Celsius are ${prediction} Fahrenheit`;
  }
}

celsius.addEventListener('input', () => {
  changeCelsiusToFahrenheit();
});

// * Load the model

let model = null;

// Automatically load the model
(async () => {
  console.info('Loading model...');
  model = await tf.loadLayersModel('./model/output/model.json');
  console.info('Model loaded!');
})();