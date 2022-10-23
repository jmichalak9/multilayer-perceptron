using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultilayerPerceptron
{
    public static class ActivationFunctions
    {
        public static IActivationFunction Sigmoid = new SigmoidClass();
        
        class SigmoidClass : IActivationFunction
        {
            public double DerivativeValue(double x)
            {
                return Value(x) * (1.0 - Value(x));
            }

            public double Value(double x)
            {
                return (double)(1.0 / (1.0 + Math.Exp(-x)));
            }
        }
    }
}
