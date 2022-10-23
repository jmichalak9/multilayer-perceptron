// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;
using System.Globalization;

class Program
{
    public static void Main()
    {
        Console.WriteLine("Training data path");
        (var trainInputsRaw, var trainLabelsRaw) = ReadDataFromFile("../../../../../data/classification/data.three_gauss.train.10000.csv");
        (var trainInputs, var trainLabels) = ProcessClassification(trainInputsRaw, trainLabelsRaw);


        Layer[] layers = {new Layer(trainInputsRaw.GetLength(1)), new Layer(5), new Layer(trainLabels.ColumnCount)};

        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.Sigmoid;

        var mlp = new MLP(layers, errorFunction, activationFunction, 0.01f, 0f);
        mlp.Fit(50, trainInputs, trainLabels);

        (var testInputsRaw, var testLabelsRaw) = ReadDataFromFile("../../../../../data/classification/data.three_gauss.test.10000.csv");
        (var testInputs, var testLabels) = ProcessClassification(testInputsRaw, testLabelsRaw);

        var predictions = mlp.Predict(testInputs);
        var accuracy = MLP.CalculateAccuracy(predictions, testLabels);
        Console.WriteLine(accuracy);
    }

    public static (Matrix<double>, Matrix<double>) ProcessClassification(double[,] inputs, double[] labels)
    {
        var classNumber = (int)labels.Cast<double>().Max();

        var outputMatrix = new double[labels.Length, classNumber];
        for (int i = 0; i < labels.Length; i++)
        {
            outputMatrix[i, (int)labels[i] - 1] = 1;
        }

        var boolInputs = Matrix<double>.Build.DenseOfArray(inputs);
        var outputLabels = Matrix<double>.Build.DenseOfArray(outputMatrix);
        return (boolInputs, outputLabels);
    }

    public static (double[,], double[]) ReadDataFromFile(string filepath)
    {
        var trainingData = File.ReadAllLines(filepath);

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
        return (inputMatrix, labels);
    }
}
