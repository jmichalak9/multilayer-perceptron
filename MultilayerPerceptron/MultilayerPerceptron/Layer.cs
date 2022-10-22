using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace MultilayerPerceptron;

public class Layer
{
    private IActivationFunction _activationFunction;
    private IErrorFunction _errorFunction;

    private Matrix<double> weights;
    private Vector<double> biases;
    private readonly int width;
    
    // backprop
    private Vector<double> lastZ;
    private Vector<double> lastActivations;
    private Vector<double> delta;

    private Layer prev;
    private Layer next;
    
    public Layer(int width)
    {
        this.width = width;
    }

    public void Initialize(Layer prev, Layer next, IActivationFunction activationFunction, IErrorFunction errorFunction, Random rng)
    {
        if (prev == null)
        {
            return;
        }

        this.prev = prev;
        this.next = next;
        biases = Vector<double>.Build.Random(width, new ContinuousUniform(0, 1));
        weights = Matrix<double>.Build.Dense(width, prev.width);

        for (int i = 0; i < weights.RowCount; i++)
        {
            for (int j = 0; j < weights.ColumnCount; j++)
            {
                weights[i, j] = rng.NextDouble();
            }
        }

        _activationFunction = activationFunction;
        _errorFunction = errorFunction;
    }

    public Vector<double> Forward(Vector<double> data)
    {
        if (prev == null)
        {
            lastActivations = data;
            return data;
        }

        lastZ = weights * data + biases;
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

        delta = (next.weights.Transpose().Multiply(next.delta)).PointwiseMultiply(lastZ.Map(_activationFunction.DerivativeValue));
    }

    public void UpdateWeight(double learningRate)
    {
        weights = weights.Subtract(learningRate * delta.OuterProduct(prev.lastActivations)); //OK
        biases -= delta;
    }
}