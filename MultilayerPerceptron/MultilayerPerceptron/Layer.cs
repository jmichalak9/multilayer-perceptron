using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Single;

namespace MultilayerPerceptron;

public class Layer
{
    private Func<Single, Single> activationFunction;
    private Func<Single, Single> activationFunctionDeriv;

    private Matrix<Single> weights;
    private Vector<Single> biases;
    private readonly int width;
    
    // backprop
    private Vector<Single> lastZ;
    private Vector<Single> lastActivations;
    private Vector<Single> lastE;
    private Matrix<Single> lastL;

    private Layer prev;
    private Layer next;

    private float learningRate;
    
    public Layer(int width)
    {
        this.width = width;
    }

    public void Initialize(float learningRate, Layer prev, Layer next)
    {
        if (prev == null)
        {
            return;
        }
        this.learningRate = learningRate;
        this.prev = prev;
        this.next = next;
        biases = Vector<Single>.Build.Random(width, new ContinuousUniform(0, 1));
        lastE = Vector<Single>.Build.Dense(width);
        lastL = Matrix<Single>.Build.Dense(width, prev.width);

        weights = Matrix<Single>.Build.Dense(width, prev.width);
        var lower = -(1.0 / Math.Sqrt(prev.width));
        var upper = (1.0 / Math.Sqrt(prev.width));
        var rand = new Random();

        for (int i = 0; i < weights.RowCount; i++)
        {
            for (int j = 0; j < weights.ColumnCount; j++)
            {
                weights[i, j] = (float)(lower + rand.NextSingle() *(upper-lower));
            }
        }

        activationFunction = f => (float)(1 / (1 + Math.Exp(-f)));
        activationFunctionDeriv = f => activationFunction(f) * (1 - activationFunction(f));
    }

    public Vector<Single> Forward(Vector<Single> data)
    {
        if (prev == null)
        {
            lastActivations = data;
            return data;
        }
        var z = Vector<Single>.Build.Dense(width);
        var activations = Vector<Single>.Build.Dense(width);
        for (int i = 0; i < width; i++)
        {
            var neuronWeights = weights.Row(i);
            var neuronBias = biases[i];
            z[i] = data * neuronWeights + neuronBias;
            activations[i] = activationFunction(z[i]);
        }

        lastZ = z;
        lastActivations = activations;
        return activations;
    }

    public void Back(Vector<Single> labels)
    {
        if (next == null)
        {
            lastE = lastActivations - labels;
        }
        else
        {
            lastE = Vector<Single>.Build.Dense(width);
            for (int i = 0; i < weights.RowCount; i++)
            {
                for (int j = 0; j < next.lastL.ColumnCount; j++)
                {
                    lastE[i] = next.lastL[j, i];

                }
            }
        }

        var dLdA = lastE * 2;
        var dAdZ = lastZ.Map(activationFunctionDeriv);
        for (int i = 0; i < weights.RowCount; i++)
        {
            lastL.SetRow(i, weights.Row(i) * lastE[i]);
        }

        for (int i = 0; i < weights.RowCount; i++)
        {
            for (int j = 0; j < weights.Row(i).Count; j++)
            {
                var dZdW = prev.lastActivations[j];
                var dLdW = dLdA[i] * dAdZ[i] * dZdW;
                weights[i, j] -= dLdW * learningRate;
            }
        }

        var biasUpdate = dLdA * dAdZ;
        biases = biases - biasUpdate * learningRate;
    }
}