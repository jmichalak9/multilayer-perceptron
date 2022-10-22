using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace MultilayerPerceptron;
public class MLP
{
    private double learningRate;
    private Random _rng;
    private Layer[] layers;
    private IErrorFunction _errorFunction;

    public MLP(Layer[] layers, IErrorFunction errorFunction, IActivationFunction activationFunction, double learningRate)
    {
        this.layers = layers;
        this.learningRate = learningRate;
        _errorFunction = errorFunction;

        _rng = new Random();
        Layer prev = null;
        // initialize
        for (int i = 0; i < layers.Length; i++)
        {
            Layer next = null;
            if (i < layers.Length - 1)
            {
                next = layers[i + 1];
            }

            layers[i].Initialize(prev, next, activationFunction, errorFunction, _rng);
            prev = layers[i];
        }
    }

    public void Fit(int epochs, Matrix<double> data, Matrix<double> labels)
    {
        for (int e = 0; e < epochs; e++)
        {
            // Randomize data order
            var randomIndices = new int[data.RowCount];
            for (int i = 0; i < randomIndices.Length; i++)
            {
                randomIndices[i] = i;
            }
            randomIndices = randomIndices.OrderBy(x => _rng.Next()).ToArray();
            data.PermuteRows(new Permutation(randomIndices));

            var predictions = Matrix<double>.Build.Dense(data.RowCount, labels.ColumnCount);
            for (int i = 0; i < data.RowCount; i++)
            {
                var activations = data.Row(i);
                foreach (var layer in layers)
                {
                    activations = layer.Forward(activations);
                }

                predictions.SetRow(i, activations);

                for (int j = layers.Length - 1; j > 0; j--)
                {
                    var layer = layers[j];
                    layer.Back(labels.Row(i));
                }

                for (int step = 1; step < layers.Length; step++)
                {
                    layers[step].UpdateWeight(learningRate);
                }
            }

            var accuracy = CalculateAccuracy(predictions, labels);
            double loss = CalculateLoss(predictions, labels);
            Console.WriteLine($"Epoch: {e}, Loss: {loss}, Accuracy: {accuracy}");
        }
    }

    private double CalculateLoss(Matrix<double> preds, Matrix<double> labels)
    {
        double result = 0;
        for (int i = 0; i < preds.RowCount; i++)
        {
            for (int j = 0; j < preds.ColumnCount; j++)
            {
                var x = preds[i, j];
                var y = labels[i, j];
                result += _errorFunction.Value(x, y);
            }
        }
        
        return result / preds.RowCount;
    }

    private double CalculateAccuracy(Matrix<double> preds, Matrix<double> labels)
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
        var predicted = Matrix<double>.Build.Dense(data.RowCount,data.ColumnCount);
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
}