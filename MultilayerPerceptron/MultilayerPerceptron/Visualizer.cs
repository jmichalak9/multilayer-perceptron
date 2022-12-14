using MathNet.Numerics.LinearAlgebra;
using OxyPlot;
using OxyPlot.Legends;
using OxyPlot.Series;
namespace MultilayerPerceptron;

public class Visualizer
{
    public static void VisualizeClassification(string name, Matrix<double> data, Vector<double> labels)
    {
        var numClasses = (int)labels.Maximum();

        var lineseries = new LineSeries[numClasses];
        for (int i = 0; i < numClasses; i++)
        {
            var line = new OxyPlot.Series.LineSeries()
            {
                Title = $"Series {i}",
                StrokeThickness = 0,
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            };
            lineseries[i] = line;
        }

        for (int i = 0; i < data.RowCount; i++)
        {
            var classNum = (int)labels[i] - 1;
            lineseries[classNum].Points.Add(new DataPoint(data[i,0], data[i,1]));
        }

        var model = new OxyPlot.PlotModel
        {
            Title = $"Classification data ({name})"
        };
        foreach (var ls in lineseries)
        {
            model.Series.Add(ls);
        }
        using (var stream = File.Create($"../../../../../{name}.pdf"))
        {
            var pdfExporter = new PdfExporter{Width = 600, Height = 400};
            pdfExporter.Export(model, stream);
        }
    }
    public static void VisualizeRegression(string name, Vector<double> x, Vector<double> y, Vector<double> predicted)
    {
        var lineseries = new ScatterSeries[]{new OxyPlot.Series.ScatterSeries()
            {
                Title = "Real",
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            }, new OxyPlot.Series.ScatterSeries()
            {
                Title = $"Predicted",
                MarkerSize = 2,
                MarkerType = OxyPlot.MarkerType.Circle
            }
        };        
        for (int i = 0; i< x.Count; i++)
        {
            lineseries[0].Points.Add(new ScatterPoint(x[i], y[i]));
            lineseries[1].Points.Add(new ScatterPoint(x[i], predicted[i]));
        }

        var model = new OxyPlot.PlotModel
        {
            Title = $"Regression data"
        };
        foreach (var ls in lineseries)
        {
            model.Series.Add(ls);
        }
        model.Legends.Add(new Legend
        {
            LegendPlacement = LegendPlacement.Outside,
            LegendPosition = LegendPosition.TopCenter,
            LegendOrientation = LegendOrientation.Horizontal,
            
        });        
        using (var stream = File.Create($"../../../../../{name}.pdf"))
        {
            var pdfExporter = new PdfExporter{Width = 600, Height = 400};
            pdfExporter.Export(model, stream);
        }
    }
}