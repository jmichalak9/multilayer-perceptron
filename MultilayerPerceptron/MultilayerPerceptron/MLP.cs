using System.Text;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace MultilayerPerceptron;
public class MLP
{
    private double learningRate;
    private double momentum;
    private Random _rng;
    private Layer[] layers;
    private IErrorFunction _errorFunction;

    private StreamWriter weightSW;
    private StreamWriter lossSW;
    public MLP(Layer[] layers, IErrorFunction errorFunction, IActivationFunction activationFunction, double learningRate, double momentum, bool withBiases, Random rng, StreamWriter weightSW, StreamWriter lossSW)
    {
        this.layers = layers;
        this.learningRate = learningRate;
        this.momentum = momentum;
        _errorFunction = errorFunction;
        this.weightSW = weightSW;
        this.lossSW = lossSW;
        _rng = rng;
        Layer prev = null;
        // initialize
        for (int i = 0; i < layers.Length; i++)
        {
            Layer next = null;
            if (i < layers.Length - 1)
            {
                next = layers[i + 1];
            }

            layers[i].Initialize(prev, next, activationFunction, errorFunction, withBiases, _rng, weightSW, lossSW, i);
            prev = layers[i];
        }
    }

    public void Fit(int epochs, Matrix<double> data, Matrix<double> labels, bool isRegression = false)
    {
        double loss = 0.0;

        var randomIndices = new int[data.RowCount];
        for (int i = 0; i < randomIndices.Length; i++)
        {
            randomIndices[i] = i;
        }

        for (int e = 0; e < epochs; e++)
        {
            // Randomize data order
            randomIndices = randomIndices.OrderBy(x => _rng.Next()).ToArray();

            var predictions = Matrix<double>.Build.Dense(data.RowCount, labels.ColumnCount);
            for (int k = 0; k < randomIndices.Length; k++)
            {
                if (k % 100 == 0)
                {
                    Console.WriteLine(k);
                }
                var currentRow = randomIndices[k];
                var activations = data.Row(currentRow);
                foreach (var layer in layers)
                {
                    activations = layer.Forward(activations);
                }

                predictions.SetRow(currentRow, activations);

                for (int j = layers.Length - 1; j > 0; j--)
                {
                    var layer = layers[j];
                    layer.Back(labels.Row(currentRow));
                }

                for (int step = 1; step < layers.Length; step++)
                {
                    layers[step].UpdateWeight(learningRate, momentum);
                }
            }
            var current_loss = CalculateLoss(predictions, labels, _errorFunction);

            if (!isRegression)
            {
                var accuracy = CalculateAccuracy(predictions, labels);
                Console.WriteLine($"Epoch: {e}, Loss: {current_loss}, dLoss: {loss - current_loss}, Accuracy: {accuracy}");
            }
            else
            {
                Console.WriteLine($"Epoch: {e}, Loss: {current_loss}, dLoss: {loss - current_loss}");
            }

            if (e % 50 == 0)
            {
                ExportWeights(e);
                ExportLoss(e);
            }
            loss = current_loss;
        }
    }

    private void ExportWeights(int epoch)
    {
        weightSW.WriteLine($"epoch {epoch}");
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ExportWeights();
        }
    }
    private void ExportLoss(int epoch)
    {
        lossSW.WriteLine($"epoch {epoch}");
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ExportLoss();
        }
    }
    public static double CalculateLoss(Matrix<double> preds, Matrix<double> labels, IErrorFunction errorFunction)
    {
        double result = 0;
        for (int i = 0; i < preds.RowCount; i++)
        {
            for (int j = 0; j < preds.ColumnCount; j++)
            {
                var x = preds[i, j];
                var y = labels[i, j];
                result += errorFunction.Value(y, x);
            }
        }
        
        return result / preds.RowCount;
    }

    public static double CalculateAccuracy(Matrix<double> preds, Matrix<double> labels)
    {
        int hit = 0;
        for (int i = 0; i < preds.RowCount; i++)
        {
            var x = preds.Row(i).MaximumIndex();
            var y = labels.Row(i).MaximumIndex();
            if (x == y)
            {
                hit++;
            }
        }

        return (double)hit / (double)preds.RowCount;
    }

    public Matrix<double> Predict(Matrix<double> data)
    {
        var predicted = Matrix<double>.Build.Dense(data.RowCount,layers[layers.Length - 1].width);
        for (int i = 0; i < data.RowCount; i++)
        {
            var activations = data.Row(i);
            foreach (var layer in layers)
            {
                activations = layer.Forward(activations);
            }

            predicted.SetRow(i, activations);
        }
        return predicted;
    }

    public static double[] PredictedClasses(Matrix<double> preds)
    {
        var classes = new double[preds.RowCount];
        for (int i = 0; i < preds.RowCount; i++)
        {
            var x = preds.Row(i).MaximumIndex();
            classes[i] = x + 1;
        }

        return classes;
    }
}