import { Component, OnInit } from '@angular/core';
import data from '../../assets/data.json';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { FormGroup, FormControl, FormArray, FormBuilder } from '@angular/forms';

@Component({
  selector: 'app-guessnumber',
  templateUrl: './guessnumber.component.html',
  styleUrls: ['./guessnumber.component.sass']
})
export class GuessnumberComponent implements OnInit {
  calculationData: Array<any>;
  settings: FormGroup;

  constructor(private fb: FormBuilder) {
    this.settings = this.fb.group({
      activation: 'relu',
      numberOfTrainingData: 400,
      numberOfLayers: 4,
      quantities: this.fb.array([]) ,
    });
  }

  addLayer() {
    let layers = this.settings.value.numberOfLayers
    this.quantities().push(this.newQuantity())


  }
  removeQuantity(i:number) {
    this.quantities().removeAt(i);
  }

  quantities() : FormArray {
    return this.settings.get("quantities") as FormArray
  }

  newQuantity(): FormGroup {
    return this.fb.group({
      units: null
    })
  }

  onSubmit(){
    console.warn(this.settings.value);
    let dataOfForm = this.settings.value;


    this.run(dataOfForm);
  }

  ngOnInit(): void {
    this.run(100)
  }
  getData(numberOfTrainingData): any {


    const cleaned = data
    return cleaned;
  }

  async run(dataOfForm) {
    // Load and plot the original input data that we are going to train on.
    const datas = await this.getData(dataOfForm.numberOfTrainingData);
    const values = datas.map(d => ({
      x: d.x,
      y: d.y,
    }));
    console.log(values);

    tfvis.render.scatterplot(
      {name: 'X'},
      {values},
      {
        xLabel: 'X',
        yLabel: 'Y',
        height: 300
      }
    );

    const model = this.createModel(dataOfForm);
    //tfvis.show.modelSummary({name: 'Model Summary'}, model);
    const tensorData = this.convertToTensor(datas);
    const {inputs, labels} = tensorData;

    // Train the model
    await this.trainModel(model, inputs, labels);
    console.log('Done Training');
    this.testModel(model, datas, tensorData);
    // More code will be added below
    await model.save('downloads://my-model');
  }

  createModel(dataOfForm): tf.Sequential {
    // Create a sequential model
    let model
    tf.tidy(() => {
      model = tf.sequential();
      // Add a single input layer
      model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
      /*for (let index = 0; index < dataOfForm.quantities.length; index++) {
        const units = dataOfForm.quantities[index].units
        model.add(tf.layers.dense({units: units}));
      }*/
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));
      model.add(tf.layers.dense({units: 50, activation: "relu6"}));


      // Add an output layer
      model.add(tf.layers.dense({units: 1, useBias: true}));

    });
    return model;
  }

  guessRandomData() {

  }

  convertToTensor(dataToConvert): any {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(dataToConvert);

      // Step 2. Convert data to Tensor
      const inputs = dataToConvert.map(d => d.x);
      const labels = dataToConvert.map(d => d.y);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  async  trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });



    const batchSize = 183;
    const epochs = 40;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 500, callbacks: ['onEpochEnd'] }
      )
    });
  }


  testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
/*
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict([xs.reshape([100, 1])]);
*/
      // const preds = model.predict([xs.reshape([1, 1])]);

      const xs = tf.linspace(0, 2, 100);
      const preds = model.predict([xs.reshape([100, 1])]);

      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      console.log(unNormXs.dataSync(), unNormPreds.dataSync())
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {

      return {x: val, y: preds[i]}
    });

    const originalPoints = inputData.map(d => ({
      x: d.x, y: d.y,
    }));


    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'},
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLabel: 'X',
        yLabel: 'Y',
        height: 300
      }
    );
  }

}
