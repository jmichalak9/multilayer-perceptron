using Cairo;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Path = System.IO.Path;

namespace MultilayerPerceptron;

public class MNIST
{
    public Matrix<double> trainInputs;
    public Matrix<double> trainLabels;
    public Matrix<double> testInputs;
    public Matrix<double> testLabels;

    public MNIST(string rootDir)
    {
        var trainInputsRaw = new IDX(Path.Combine(rootDir, "train-images-idx3-ubyte.gz"));
        var trainLabelsRaw = new IDX(Path.Combine(rootDir, "train-labels-idx1-ubyte.gz"));
        var testInputsRaw = new IDX(Path.Combine(rootDir, "t10k-images-idx3-ubyte.gz"));
        var testLabelsRaw = new IDX(Path.Combine(rootDir, "t10k-labels-idx1-ubyte.gz"));
        
        int numClasses = 10;
        trainInputs = Matrix<double>.Build.Dense(trainInputsRaw.dim[0], trainInputsRaw.dim[1]*trainInputsRaw.dim[2]);
        trainLabels = Matrix<double>.Build.Dense(trainLabelsRaw.dim[0], numClasses);
        testInputs = Matrix<double>.Build.Dense(testInputsRaw.dim[0], testInputsRaw.dim[1]*testInputsRaw.dim[2]);
        testLabels = Matrix<double>.Build.Dense(testLabelsRaw.dim[0], numClasses);

        for (int i = 0; i < trainInputs.RowCount; i++)
        {
            for (int j = 0; j < trainInputs.ColumnCount; j++)
            {
                var idx = trainInputs.ColumnCount * i + j;
                trainInputs[i, j] = trainInputsRaw.data[idx] / 255.0;
            }
        }

        for (int i = 0; i < testInputs.RowCount; i++)
        {
            for (int j = 0; j < testInputs.ColumnCount; j++)
            {
                var idx = testInputs.ColumnCount * i + j;
                testInputs[i, j] = testInputsRaw.data[idx] / 255.0;
            }
        }
        
        for (int i = 0; i < trainLabelsRaw.data.Length; i++)
        {
            trainLabels[i,trainLabelsRaw.data[i]] = 1;
        }

        for (int i = 0; i < testLabelsRaw.data.Length; i++)
        {
            testLabels[i,testLabelsRaw.data[i]] = 1;
        }
    }
}