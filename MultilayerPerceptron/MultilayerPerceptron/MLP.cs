using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using MathNet.Numerics.LinearAlgebra;

namespace MultilayerPerceptron;
public class MLP
{
    private float learningRate;
    private Layer[] layers;

    public MLP(Layer[] layers, float learningRate)
    {
        this.layers = layers;
        this.learningRate = learningRate;
        
        Layer prev = null;
        // initialize
        for (int i = 0; i < layers.Length; i++)
        {
            Layer next = null;
            if (i < layers.Length - 1)
            {
                next = layers[i + 1];
            }

            layers[i].Initialize(this.learningRate, prev, next);
            prev = layers[i];
        }
    }

    public float Fit(int epochs, Matrix<Single> data, Matrix<Single> labels)
    {
        float loss = 1;
        for (int e = 0; e < epochs; e++)
        {
            var predictions = Matrix<Single>.Build.Dense(data.RowCount, labels.ColumnCount);
            for (int i = 0; i < data.RowCount; i++)
            {
                var activations = data.Row(i);
                foreach (var layer in layers)
                {
                    activations = layer.Forward(activations);
                    
                }
                predictions.SetRow(i, activations);

                for (int step = 0; step < layers.Length; step++)
                {
                    var l = layers.Length - (step + 1);
                    var layer = layers[l];
                    if (l == 0)
                    {
                        continue;
                    }
                    layer.Back(labels.Row(i));
                }
            }

            loss = calculateLoss(predictions, labels);
            Console.WriteLine($"{e} {loss}");
        }

        return loss;
    }

    private float calculateLoss(Matrix<Single> preds, Matrix<Single> labels)
    {
        float squared = 0;
        int count = 0;
        for (int i = 0; i < preds.RowCount; i++)
        {
            for (int j = 0; j < preds.ColumnCount; j++)
            {
                var rand = new Random();
                var x = preds[i, j];
                var y = labels[i, j];
                count++;
                squared += (x - y) * (x - y);
            }
        }
        
        return squared/ count;
    }

    public Matrix<Single> Predict(Matrix<Single> data)
    {
        var predicted = Matrix<Single>.Build.Dense(data.RowCount,data.ColumnCount);
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

    private void ValidateInputs()
    {
        
    }
}