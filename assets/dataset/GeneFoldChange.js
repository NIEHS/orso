import React from 'react';


class GeneFoldChange extends React.Component {

    drawPlotly(){
        var genes = [], changes = [];
        for(var i = 0; i < this.props.data.length; i++){
            genes.push(this.props.data[i][0]);
            changes.push(this.props.data[i][1]);
        }

        var data = [{
            x: genes,
            y: changes,
            type: 'bar',
        }];

        Plotly.newPlot('fold_change_plot', data);
    }

    componentDidMount(){
        this.drawPlotly();
    }

    componentWillUnmount(){
        $(this.refs.fold_change_plot).clear();
    }

    render(){
        return <div ref='fold_change_plot' id='fold_change_plot'></div>;
    }
}

GeneFoldChange.propTypes = {
    data: React.PropTypes.array.isRequired,
};

export default GeneFoldChange;
