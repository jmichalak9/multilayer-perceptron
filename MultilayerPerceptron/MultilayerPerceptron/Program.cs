// See https://aka.ms/new-console-template for more information
using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;
using System.Globalization;
using System.Text;

class Program
{
    private static int seed = 48;
    public static void Main()
    {
        //Classification();
        //Regression();
        MNIST();
    }

    public static void Regression()
    {
        (var trainInputsRaw, var trainLabelsRaw) = ReadDataFromFile("../../../../../data_2/regression/data.multimodal.train.100.csv");

        var outputMatrix = new double[trainLabelsRaw.Length, 1];
        for (int i = 0; i < trainLabelsRaw.Length; i++)
        {
            outputMatrix[i, 0] = trainLabelsRaw[i];
        }

        var trainInputs = Matrix<double>.Build.DenseOfArray(trainInputsRaw);
        var trainOutput = Matrix<double>.Build.DenseOfArray(outputMatrix);
        
        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.Linear(0.1);

        Layer[] layers = { new Layer(trainInputsRaw.GetLength(1)), new Layer(10), new Layer(1, ActivationFunctions.Linear(0.1)) };
        var weightSW = new StreamWriter("../../../../../weights.txt");
        var lossSW = new StreamWriter("../../../../../loss.txt");

        var mlp = new MLP(layers, errorFunction, activationFunction, 0.0001f, 0f, true, new Random(seed), weightSW, lossSW);
        mlp.Fit(500, trainInputs, trainOutput, true);

        weightSW.Close();
        lossSW.Close();
        
        var train_predictions = mlp.Predict(trainInputs);

        (var testInputsRaw, var testLabelsRaw) = ReadDataFromFile("../../../../../data_2/regression/data.multimodal.test.100.csv");

        outputMatrix = new double[testLabelsRaw.Length, 1];
        for (int i = 0; i < testLabelsRaw.Length; i++)
        {
            outputMatrix[i, 0] = testLabelsRaw[i];
        }

        var testInputs = Matrix<double>.Build.DenseOfArray(testInputsRaw);
        var testOutput = Matrix<double>.Build.DenseOfArray(outputMatrix);

        var test_predictions = mlp.Predict(testInputs);
        var loss = MLP.CalculateLoss(test_predictions, testOutput, ErrorFunctions.Square);
        Console.WriteLine(loss);
        Visualizer.VisualizeRegression("regression_train", trainInputs.Column(0), trainOutput.Column(0), train_predictions.Column(0));
        Visualizer.VisualizeRegression("regression_test", testInputs.Column(0), testOutput.Column(0), test_predictions.Column(0));
    }

    public static void MNIST()
    {
        var mnist = new MNIST("../../../../../data/mnist");
        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.Sigmoid;

        var weightSW = new StreamWriter("../../../../../weights.txt");
        var lossSW = new StreamWriter("../../../../../loss.txt");

        Layer[] layers = { new Layer(mnist.trainInputs.ColumnCount), new Layer(64), new Layer(mnist.trainLabels.ColumnCount) };
        var mlp = new MLP(layers, errorFunction, activationFunction, 0.1f, 0f, true, new Random(seed), weightSW, lossSW);
        mlp.Fit(1000, mnist.trainInputs, mnist.trainLabels);
        weightSW.Close();
        lossSW.Close();
        
        var predictions = mlp.Predict(mnist.testInputs);
        var accuracy = MLP.CalculateAccuracy(predictions, mnist.testLabels);
        Console.WriteLine(accuracy);

    }

    public static void Classification()
    {

        Console.WriteLine("Training data path");
        (var trainInputsRaw, var trainLabelsRaw) = ReadDataFromFile("../../../../../data/projekt1-oddanie/clasification/data.noisyXOR.train.1000.csv");
        (var trainInputs, var trainLabels) = ProcessClassification(trainInputsRaw, trainLabelsRaw);

        Layer[] layers = { new Layer(trainInputsRaw.GetLength(1)), new Layer(2), new Layer(trainLabels.ColumnCount) };

        var errorFunction = ErrorFunctions.Square;
        var activationFunction = ActivationFunctions.Tanh;

        var weightSW = new StreamWriter("../../../../../weights.txt");
        var lossSW = new StreamWriter("../../../../../loss.txt");

        var mlp = new MLP(layers, errorFunction, activationFunction, 0.1f, 0f, true, new Random(seed), weightSW, lossSW);
        mlp.Fit(1000, trainInputs, trainLabels);
        weightSW.Close();
        lossSW.Close();

        (var testInputsRaw, var testLabelsRaw) = ReadDataFromFile("../../../../../data/projekt1-oddanie/clasification/data.noisyXOR.test.1000.csv");
        (var testInputs, var testLabels) = ProcessClassification(testInputsRaw, testLabelsRaw);

        var predictions = mlp.Predict(testInputs);
        var accuracy = MLP.CalculateAccuracy(predictions, testLabels);
        Console.WriteLine(accuracy);
        Visualizer.VisualizeClassification("classification_test", testInputs, Vector<double>.Build.DenseOfArray(testLabelsRaw));
        Visualizer.VisualizeClassification("classification_predictions", testInputs, Vector<double>.Build.DenseOfArray(MLP.PredictedClasses(predictions)));
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
