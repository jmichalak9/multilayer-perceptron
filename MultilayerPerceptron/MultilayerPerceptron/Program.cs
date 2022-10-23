// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;
using System.Globalization;
using System.Reflection.Emit;

class Program
{
    private static int seed = 69;
    public static void Main()
    {
        Classification();
        //Regression();
    }

    public static void Regression()
    {
        Console.WriteLine("Training data path");
        (var trainInputsRaw, var trainLabelsRaw) = ReadDataFromFile("../../../../../data/regression/data.cube.train.1000.csv");

        var outputMatrix = new double[trainLabelsRaw.Length, 1];
        for (int i = 0; i < trainLabelsRaw.Length; i++)
        {
            outputMatrix[i, 0] = trainLabelsRaw[i];
        }

        var trainInputs = Matrix<double>.Build.DenseOfArray(trainInputsRaw);
        var trainOutput = Matrix<double>.Build.DenseOfArray(outputMatrix);

        Layer[] layers = { new Layer(trainInputsRaw.GetLength(1)), new Layer(5), new Layer(1) };

        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.ReLU;

        var mlp = new MLP(layers, errorFunction, activationFunction, 0.001f, 0f, true, new Random(seed));
        mlp.Fit(200, trainInputs, trainOutput, true);

        (var testInputsRaw, var testLabelsRaw) = ReadDataFromFile("../../../../../data/regression/data.cube.test.1000.csv");
        outputMatrix = new double[testLabelsRaw.Length, 1];
        for (int i = 0; i < trainLabelsRaw.Length; i++)
        {
            outputMatrix[i, 0] = testLabelsRaw[i];
        }

        var testInputs = Matrix<double>.Build.DenseOfArray(testInputsRaw);
        var testOutput = Matrix<double>.Build.DenseOfArray(outputMatrix);

        var predictions = mlp.Predict(testInputs);
        var loss = MLP.CalculateLoss(predictions, testOutput, ErrorFunctions.Square);
        Console.WriteLine(loss);
        Visualizer.VisualizeRegression("activation", testInputs.Column(0), testOutput.Column(0), predictions.Column(0));
    }

    public static void Classification()
    {
        Console.WriteLine("Training data path");
        (var trainInputsRaw, var trainLabelsRaw) = ReadDataFromFile("../../../../../data/classification/data.three_gauss.train.1000.csv");
        (var trainInputs, var trainLabels) = ProcessClassification(trainInputsRaw, trainLabelsRaw);

        Layer[] layers = { new Layer(trainInputsRaw.GetLength(1)), new Layer(5), new Layer(trainLabels.ColumnCount) };

        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.Sigmoid;


        var mlp = new MLP(layers, errorFunction, activationFunction, 0.001f, 0f, true, new Random(seed));
        mlp.Fit(250, trainInputs, trainLabels);

        (var testInputsRaw, var testLabelsRaw) = ReadDataFromFile("../../../../../data/classification/data.three_gauss.test.1000.csv");
        (var testInputs, var testLabels) = ProcessClassification(testInputsRaw, testLabelsRaw);

        var predictions = mlp.Predict(testInputs);
        var accuracy = MLP.CalculateAccuracy(predictions, testLabels);
        Console.WriteLine(accuracy);
        Visualizer.VisualizeClassification("test", testInputs, Vector<double>.Build.DenseOfArray(testLabelsRaw));
        Visualizer.VisualizeClassification("predictions", testInputs, Vector<double>.Build.DenseOfArray(MLP.PredictedClasses(predictions)));
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
