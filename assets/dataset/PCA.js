import React from 'react';


class PCA extends React.Component {

    constructor(props) {
        super(props);

        let color_by_choices = ['None'];
        for (let key in this.props.data['attributes']) {
            color_by_choices.push(key);
        }

        this.state = {
            color_by: 'None',
            color_by_choices: color_by_choices,
        };
    }

    drawD3(svgElement, data){
        var margins = {
                left: 40,
                bottom: 60,
                top: 10,
                right: 10,
            },
            h = $(svgElement).height() - margins.bottom - margins.top,
            w = $(svgElement).width() - margins.left - margins.right;

        var pca_1 = [], pca_2 = [];
        for (var i in data['pca']) {
            pca_1.push(data['pca'][i][0]);
            pca_2.push(data['pca'][i][1]);
        }

        var palette = d3.scale.category20(),
            palette_length = 20;
        var colors = [];
        if (this.state.color_by == 'None') {
            for (var i in data['pca']) {
                colors.push(0);
            }
        } else {
            let list = data['attributes'][this.state.color_by];
            let used = [];
            for (var i in list) {
                if ( $.inArray(list[i], used) == -1) {
                    used.push(list[i]);
                }
                // console.log($.inArray(list[i], used));
                // console.log($.inArray(list[i], used) % palette.length);
                colors.push($.inArray(list[i], used) % palette_length);
            }
        }

        var x = d3.scale.linear()
            .domain([Math.min.apply(Math,pca_1), Math.max.apply(Math,pca_1)])
            .range([margins.left, w + margins.left]);

        var y = d3.scale.linear()
            .domain([Math.min.apply(Math,pca_2), Math.max.apply(Math,pca_2)])
            .range([h, margins.top]);

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
            .data(data['pca'])
            .enter()
            .append('circle')
            .attr('r', 3)
            .attr('cx', function(d) {return x(d[0]);})
            .attr('cy', function(d) {return y(d[1]);})
            .attr('fill', function(d, i) {return palette(colors[i]);});
    }

    removeD3(svgElement){
        $(svgElement).empty();
    }

    componentDidMount(){
        let svg = this.refs.svg;
        this.drawD3(svg, this.props.data);

        var $color_by_select = $(this.refs.color_by_select);
        for (let i in this.state.color_by_choices) {
            $color_by_select.append(
                '<option val="' + i + '">' + this.state.color_by_choices[i] + '</option>');
        }
    }

    componentDidUpdate(){
        let svg = this.refs.svg;
        this.removeD3(this.refs.svg);
        this.drawD3(svg, this.props.data);
    }

    componentWillUnmount(){
        this.removeD3(this.refs.svg);
    }

    change(event){
         this.setState({color_by: event.target.value});
     }

    render(){
        console.log(this.props.data);
        console.log(this.state);

        return <div className='pca'>
            <div className='row'>
                <div className='col-sm-8'>
                    <svg style={{height:"100%", width:"100%"}} ref='svg'></svg>
                </div>
                <div className='col-sm-4'>
                    <div>Color by</div>
                    <select ref='color_by_select'
                        onChange={this.change.bind(this)}
                        value={this.state.color_by}>
                    </select>
                </div>
            </div>
        </div>;
    }
}

PCA.propTypes = {
    data: React.PropTypes.object.isRequired,
};

export default PCA;
