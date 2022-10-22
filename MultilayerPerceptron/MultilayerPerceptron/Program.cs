// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;
using System.Globalization;

Console.WriteLine("Training data path");
var trainingPath = Console.ReadLine();

var trainingData = File.ReadAllLines(trainingPath);

// Skip headers
trainingData = trainingData.Skip(1).ToArray();

var dataLength = trainingData.Length;
var inputSize = trainingData.First().Split(',').Length;
var inputMatrix = new double[dataLength, inputSize];
var labels = new double[dataLength];
for (int j = 0; j < dataLength; j++)
{
    var line = trainingData[j];

    var numbers = line.Split(',');
    for (int i = 0; i < numbers.Length; i++)
    {
        if (i == numbers.Length - 1)
        {
            inputMatrix[j, i] = 1; //bias
            labels[j] = double.Parse(numbers[i], CultureInfo.InvariantCulture.NumberFormat);
            break;
        }

        inputMatrix[j, i] = double.Parse(numbers[i], CultureInfo.InvariantCulture.NumberFormat);
    }
}

var classNumber = (int)labels.Cast<double>().Max();

var outputMatrix = new double[dataLength, classNumber];
for (int i = 0; i < dataLength; i++)
{
    outputMatrix[i, (int)labels[i] - 1] = 1;
}

var boolInputs = Matrix<double>.Build.DenseOfArray(inputMatrix);
var outputLabels = Matrix<double>.Build.DenseOfArray(outputMatrix);

Layer[] layers = {new Layer(inputSize), new Layer(inputSize + 10), new Layer(2)};

var errorFunction = ErrorFunctions.Square;
var activationFunction = ActivationFunctions.Sigmoid;

var mlp = new MLP(layers, errorFunction, activationFunction, 0.1f);

mlp.Fit(5000, boolInputs, outputLabels);