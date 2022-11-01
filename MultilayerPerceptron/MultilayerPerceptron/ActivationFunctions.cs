using MathNet.Numerics.LinearAlgebra;
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
        public static SoftMaxClass SoftMax = new SoftMaxClass();
        public static IActivationFunction LeakyReLU(double a) => new LeakyReLUClass(a);
        public static IActivationFunction Linear(double a) => new LinearClass(a);

        public class SoftMaxClass
        {
            public Matrix<double> DerivativeValue(Vector<double> z)
            {
                var s = Value(z);

                var si_sj = -s.OuterProduct(s);
                var result = Matrix<double>.Build.DenseOfDiagonalVector(s);
                return result + si_sj;
            }

            public Vector<double> Value(Vector<double> z)
            {
                var result = new double[z.Count];
                var max = z.Max();
                z = z.Map(x => Math.Exp(x - max));
                var sum = z.Sum();

                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = z[i] / sum;
                }

                return Vector<double>.Build.DenseOfArray(result);
            }
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

        class LinearClass : IActivationFunction
        {
            private double a;

            public LinearClass(double a)
            {
                this.a = a;
            }

            public double DerivativeValue(double x)
            {
                return a;
            }

            public double Value(double x)
            {
                return a * x;
            }
        }
    }
}
