namespace MultilayerPerceptron
{
    public interface IActivationFunction
    {
        double Value(double x);
        double DerivativeValue(double x);
    }
}
