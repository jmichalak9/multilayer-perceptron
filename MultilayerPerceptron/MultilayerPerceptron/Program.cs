// See https://aka.ms/new-console-template for more information

using System.Globalization;
using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;

var trainingData = File.ReadAllLines("../../../../../data/classification/data.three_gauss.train.10000.csv");
trainingData = trainingData.Skip(1).ToArray();

var dataLength = trainingData.Length;
var inputSize = trainingData.First().Split(',').Length;
var inputMatrix = new float[dataLength, inputSize];
var labels = new float[dataLength];
for (int j = 0; j < dataLength; j++)
{
    var line = trainingData[j];

    var numbers = line.Split(',');
    for (int i = 0; i < numbers.Length; i++)
    {
        if (i == numbers.Length - 1)
        {
            inputMatrix[j, i] = 1; //bias
            labels[j] = float.Parse(numbers[i], CultureInfo.InvariantCulture.NumberFormat);
            break;
        }

        inputMatrix[j, i] = float.Parse(numbers[i], CultureInfo.InvariantCulture.NumberFormat);
    }
}

var classNumber = (int)labels.Cast<float>().Max();
var outputMatrix = new float[dataLength, classNumber];
for (int i = 0; i < dataLength; i++)
{
    outputMatrix[i, (int)labels[i] - 1] = 1;
}

var boolInputs = Matrix<Single>.Build.DenseOfArray(inputMatrix);
var outputLabels = Matrix<Single>.Build.DenseOfArray(outputMatrix);

Layer[] layers = {new Layer(inputSize), new Layer(inputSize), new Layer(classNumber)};
var mlp = new MLP(layers, 0.1f);


var loss = mlp.Fit(100, boolInputs, outputLabels);
Console.WriteLine(loss);