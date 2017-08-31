import React from 'react';


class Expression extends React.Component {

    drawPlotly(){
        var labels = [],
            medians = [],
            mins = [],
            maxs = [];
        for(var i = 0; i < this.props.data.length; i++){
            labels.push(this.props.data[i]['cell_type']);

            var median = this.props.data[i]['median'],
                min = this.props.data[i]['min'],
                max = this.props.data[i]['max'];

            medians.push(median);
            mins.push(median - min);
            maxs.push(max - median);
        }

        var data = [{
            x: labels.reverse(),
            y: medians.reverse(),
            error_y: {
                type: 'data',
                symmetric: false,
                array: maxs.reverse(),
                arrayminus: mins.reverse(),
            },
            type: 'bar',
        }];

        Plotly.newPlot('expression_plot', data);
    }

    componentDidMount(){
        this.drawPlotly();
    }

    componentWillUnmount(){
        $('#expression_plot').clear();
    }

    render(){
        return <div id='expression_plot'></div>;
    }
}

Expression.propTypes = {
    data: React.PropTypes.array.isRequired,
};

export default Expression;
