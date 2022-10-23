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
        public static IActivationFunction Tanh = new TanhClass();
        public static IActivationFunction ReLU = new ReLUClass();

        public static IActivationFunction LeakyReLU(double a)
        {
           return new LeakyReLUClass(a);
        }

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
        class TanhClass : IActivationFunction
        {
            public double DerivativeValue(double x)
            {
                return 1 - Math.Pow(Math.Tanh(x), 2);
            }

            public double Value(double x)
            {
                return Math.Tanh(x);
            }
        }
        class ReLUClass : IActivationFunction
        {
            public double DerivativeValue(double x)
            {
                return x > 0 ? 1 : 0;
            }

            public double Value(double x)
            {
                return Math.Max(0, x);
            }
        }
        class LeakyReLUClass : IActivationFunction
        {
            private double a;
            public LeakyReLUClass(double a)
            {
                this.a = a;
            }
            public double DerivativeValue(double x)
            {
                return x > 0 ? 1 : a;
            }

            public double Value(double x)
            {
                return Math.Max(a*x, x);
            }
        }
    }
}
