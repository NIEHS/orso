import React from 'react';

import './IntersectionComparison.css';


class IntersectionComparison extends React.Component {

    drawD3(svgElement){
        var x_data = this.props.x_data,
            y_data = this.props.y_data,
            margins = {
                left: 100,
                bottom: 60,
                top: 10,
                right: 40,
            },
            h = $(svgElement).height() - margins.bottom - margins.top,
            w = $(svgElement).width() - margins.left - margins.right;

        var scatter = [];
        for (var i = 0; i < x_data.length; i++) {
            scatter.push([x_data[i], y_data[i]]);
        }

        var x = d3.scale.linear()
            .domain([1,d3.max(x_data)])
            .range([margins.left, w + margins.left]);

        var y = d3.scale.linear()
            .domain([1,d3.max(y_data)])
            .range([h, margins.top])
            .nice()
            .clamp(true);

        var xAxis = d3.svg.axis()
            .scale(x)
            .ticks(3)
            .orient('bottom');

        var yAxis = d3.svg.axis()
            .scale(y)
            .ticks(4)
            .orient('left');

        d3.select(svgElement).append('svg:g')
            .attr('class', 'x axis')
            .attr('transform', `translate(0,${($(svgElement).height() - margins.top - margins.bottom)})`)
            .call(xAxis);

        d3.select(svgElement).append('svg:g')
            .attr('class', 'y axis')
            .attr('transform', `translate(${margins.left},0)`)
            .call(yAxis);

        d3.select(svgElement).append('svg:g')
            .selectAll('circle')
            .data(scatter)
            .enter()
            .append('circle')
            .attr('r', 3.5)
            .attr('cx', function(d) {return x(d[0]);})
            .attr('cy', function(d) {return y(d[1]);});
    }

    removeD3(svgElement){
        $(svgElement).empty();
    }

    componentDidMount(){
        this.drawD3(this.refs.svg);
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    render(){
        return <div className='intersection_comparison'>
            <svg style={{height:"100%", width:"100%"}} ref='svg'></svg>
        </div>;
    }
}

IntersectionComparison.propTypes = {
    x_name: React.PropTypes.string,
    y_name: React.PropTypes.string,
    x_data: React.PropTypes.array.isRequired,
    y_data: React.PropTypes.array.isRequired,
};

export default IntersectionComparison;
