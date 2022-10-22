using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MultilayerPerceptron
{
    public interface IErrorFunction
    {
        double Value(double x, double y);

        double DerivativeValue(double x, double y);
    }
}
