using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultilayerPerceptron
{
    public static class ErrorFunctions
    {
        public static IErrorFunction Square = new SquareClass();

        class SquareClass : IErrorFunction
        {
            public double DerivativeValue(double x, double y)
            {
                return x - y;
            }

            public double Value(double x, double y)
            {
                return (x - y) * (x - y) / 2f;
            }
        }
    }
}
