﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultilayerPerceptron
{
    public static class ErrorFunctions
    {
        public static IErrorFunction Square = new SquareClass();

        public static IErrorFunction PseudoHuber(double delta)
        {
            return new PseudoHuberClass(delta);
        } 

        class SquareClass : IErrorFunction
        {
            public double DerivativeValue(double x, double y)
            {
                return -(x - y);
            }

            public double Value(double x, double y)
            {
                return (x - y) * (x - y) / 2.0;
            }
        }
        class PseudoHuberClass : IErrorFunction
        {
            private double delta;

            public PseudoHuberClass(double delta)
            {
                this.delta = delta;
            }
            public double DerivativeValue(double x, double y)
            {
                var z = x - y;
                if (Math.Abs(z) <= delta)
                {
                    return -z;
                }

                return -delta * Math.Sign(z);
            }

            public double Value(double x, double y)
            {
                var z = x - y;
                if (Math.Abs(z) <= delta)
                {
                    return z * z / 2;
                }
                return delta* z - delta * delta / 2;
            }
        }
    }
}
