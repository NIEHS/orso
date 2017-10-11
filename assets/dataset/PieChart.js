import React from 'react';

import './PieChart.css';


class PieChart extends React.Component {

    drawPie(){
        var data = [{
            values: Object.values(this.props.data),
            labels: Object.keys(this.props.data),
            type: 'pie',
            hoverinfo: 'none',
        }];

        var layout = {
            height: $(this.refs.pie).width(),
            margin: {
                l: 10,
                r: 10,
                b: 10,
                t: 10,
                pad: 4,
            },
        };

        var options = {
            displaylogo: false,
            displayModeBar: false,
        };

        Plotly.newPlot(this.pieId(), data, layout, options);
    }

    clearPie(){
        $(this.refs.pie).empty();
    }

    componentDidMount(){
        this.drawPie();
    }

    componentWillUnmount(){
        this.clearPie();
    }

    pieId(){
        return '_pie_' + this.props.index;
    }

    render(){
        return <div ref='pie' id={this.pieId()}></div>;
    }
}

PieChart.defaultProps = {
    index: 0,
};

PieChart.propTypes = {
    data: React.PropTypes.object.isRequired,
    index: React.PropTypes.number.isRequired,
};

export default PieChart;
