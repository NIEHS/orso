import React from 'react';


class GeneScatter extends React.Component {

    drawPlotly(){
        var genes = [], x = [], y = [];
        for(var i = 0; i < this.props.data.length; i++){
            genes.push(this.props.data[i][0]);
            x.push(this.props.data[i][1]);
            y.push(this.props.data[i][2]);
        }

        var data = [{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: genes,
        }];

        var layout = {
            xaxis: {
                type: 'log',
                autorange: true,
            },
            yaxis: {
                type: 'log',
                autorange: true,
            },
        };

        Plotly.newPlot('gene_scatter_plot', data, layout);
    }

    componentDidMount(){
        this.drawPlotly();
    }

    componentWillUnmount(){
        $(this.refs.gene_scatter_plot).clear();
    }

    render(){
        return <div ref='gene_scatter_plot' id='gene_scatter_plot'></div>;
    }
}

GeneScatter.propTypes = {
    data: React.PropTypes.array.isRequired,
};

export default GeneScatter;
