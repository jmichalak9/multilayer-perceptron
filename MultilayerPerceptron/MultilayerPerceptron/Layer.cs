using System.Text;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace MultilayerPerceptron;

public class Layer
{
    private IActivationFunction _activationFunction;
    private IErrorFunction _errorFunction;

    private Matrix<double> weights;
    Matrix<double> weightsChange;
    private Vector<double> biases;
    private bool withBiases;
    public readonly int width;
    
    // backprop
    private Vector<double> lastZ;
    private Vector<double> lastActivations;
    private Vector<double> delta;

    private Layer prev;
    private Layer next;
    private StreamWriter weightSW;
    private StreamWriter lossSW;
    private int layerNum;

    
    public Layer(int width, IActivationFunction activationFunction = null )
    {
        this.width = width;
        _activationFunction = activationFunction;
    }

    public void Initialize(Layer prev, Layer next, IActivationFunction activationFunction, IErrorFunction errorFunction, bool withBiases, Random rng, StreamWriter weightSW, StreamWriter lossSW, int layerNum)
    {
        if (prev == null)
        {
            return;
        }

        this.prev = prev;
        this.next = next;
        this.layerNum = layerNum;
        this.weightSW = weightSW;
        this.lossSW = lossSW;
        this.withBiases = withBiases;
        biases = Vector<double>.Build.Random(width, new ContinuousUniform(0, 1, rng));
        weights = Matrix<double>.Build.Dense(width, prev.width);

        for (int i = 0; i < weights.RowCount; i++)
        {
            for (int j = 0; j < weights.ColumnCount; j++)
            {
                weights[i, j] = rng.NextDouble() * 0.5;
            }
        }

        if(_activationFunction == null)
        {
            _activationFunction = activationFunction;
        }
        _errorFunction = errorFunction;
    }

    public Vector<double> Forward(Vector<double> data)
    {
        if (prev == null)
        {
            lastActivations = data;
            return data;
        }

        if (withBiases)
        {
            lastZ = weights * data + biases;
        }
        else
        {
            lastZ = weights * data;
        }
        lastActivations = lastZ.Map(_activationFunction.Value);

        return lastActivations;
    }

    public void Back(Vector<double> labels)
    {
        if (next == null)
        {
            var errors = new double[labels.Count];
            for (int i = 0; i < labels.Count; i++)
            {
                errors[i] = _errorFunction.DerivativeValue(labels[i], lastActivations[i]);
            }

            var dE = Vector<double>.Build.DenseOfArray(errors);
            delta = dE.PointwiseMultiply(lastZ.Map(_activationFunction.DerivativeValue)); //OK
            return;
        }

        delta = (next.delta * next.weights).PointwiseMultiply(lastZ.Map(_activationFunction.DerivativeValue));
    }

    public void ExportWeights()
    {
        if (weights == null)
            return;
        weightSW.WriteLine($"layer {layerNum}: ");
        for (int j = 0; j < weights.RowCount; j++)
        {
            for (int k = 0; k < weights.ColumnCount; k++)
                weightSW.Write($"{weights[j,k]} ");
            weightSW.WriteLine();
        }
    }
    public void ExportLoss()
    {
        if (weights == null)
            return;
        lossSW.WriteLine($"layer {layerNum}: ");
        for (int j = 0; j < weightsChange.RowCount; j++)
        {
            for (int k = 0; k < weightsChange.ColumnCount; k++)
                lossSW.Write($"{weightsChange[j,k]} ");
            lossSW.WriteLine();
        }
    }
    public void UpdateWeight(double learningRate, double momentum)
    {
        if (weightsChange == null)
        {
            weightsChange = -learningRate * delta.OuterProduct(prev.lastActivations);
        }
        else
        {
            weightsChange = -learningRate * delta.OuterProduct(prev.lastActivations) + momentum * weightsChange;
        }
        weights += weightsChange; //OK
        if (withBiases)
        {
            biases -= delta;
        }
    }
}