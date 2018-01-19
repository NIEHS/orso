import React from 'react';


class RecommendationScatter extends React.Component {

    constructor(props) {
        super(props);

        var exp_type_choices = (['--']).concat(Object.keys(this.props.data['paired_data']));

        this.state = {
            exp_type: '--',
            exp_type_choices: exp_type_choices,
        };
    }

    drawPlotlyScatter(){
        var x = [], y = [], names = [];

        if (this.state.exp_type != '--') {
            var _data = this.props.data['paired_data'][this.state.exp_type];
            for(var i = 0; i < _data.length; i++){
                names.push(_data[i][0] + ':' + _data[i][1])
                x.push(_data[i][2]);
                y.push(_data[i][3]);
            }
        }

        var plot_data = [{
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: names,
        }];

        var layout = {
            xaxis: {
                autorange: true,
            },
            yaxis: {
                autorange: true,
            },
        };

        Plotly.newPlot('rec_scatter_plot', plot_data, layout);
    }

    drawPlotlyBoxPlot(){
        var plot_data = [];

        if (this.state.exp_type != '--') {
            var _data = this.props.data['quartiled_data'][this.state.exp_type];
            for(var i = 0; i < _data.length; i++){
                plot_data.push({
                    y: _data[i],
                    type: 'box',
                    boxpoints: 'all',
                    jitter: 0.3,
                    pointpos: -1.8,
                });
            }
        }

        Plotly.newPlot('rec_box_plot', plot_data);
    }

    drawPlotly(){
        this.drawPlotlyScatter();
        this.drawPlotlyBoxPlot();
    }

    componentDidMount(){
        this.drawPlotly();

        for (let i in this.state.exp_type_choices) {
            $(this.refs.exp_type_select).append(
                '<option val="' + i + '">' + this.state.exp_type_choices[i] + '</option>');
        }
    }

    componentWillUnmount(){
        $(this.refs.gene_scatter_plot).clear();
    }

    update_exp_type(event){
        this.setState({
            exp_type: event.target.value,
        }, this.drawPlotly);
    }

    render(){
        return <div>
            <select ref='exp_type_select'
                onChange={this.update_exp_type.bind(this)}
                value={this.state.exp_type}>
            </select>
            <div ref='rec_scatter_plot' id='rec_scatter_plot'></div>
            <div ref='rec_box_plot' id='rec_box_plot'></div>
        </div>;
    }
}

RecommendationScatter.propTypes = {
    data: React.PropTypes.object.isRequired,
};

export default RecommendationScatter;
