import React from 'react';


class GeneValues extends React.Component {

    drawPlotly(){
        var genes = [], values = [];
        for(var i = 0; i < this.props.data.length; i++){
            genes.push(this.props.data[i][0]);
            values.push(this.props.data[i][1]);
        }

        var data = [{
            x: genes,
            y: values,
            type: 'bar',
        }];

        var layout = {
            margin: {
                l: 50,
                r: 50,
                b: 80,
                t: 40,
                pad: 4,
            }
        };

        var options = {
        };

        Plotly.newPlot('gene_values_plot', data, layout, options);
    }

    componentDidMount(){
        this.drawPlotly();
    }

    componentWillUnmount(){
        $(this.refs.gene_values_plot).clear();
    }

    render(){
        return <div ref='gene_values_plot' id='gene_values_plot'></div>;
    }
}

GeneValues.propTypes = {
    data: React.PropTypes.array.isRequired,
};

export default GeneValues;
