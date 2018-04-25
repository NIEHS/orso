import React from 'react';


class Network extends React.Component {

    drawNetwork(){
        var nodes = new vis.DataSet(this.props.network['nodes']);
        var edges = new vis.DataSet(this.props.network['edges']);
        var data = {
            nodes: nodes,
            edges: edges,
        };

        var options = {};

        var network = new vis.Network($(this.refs.network)[0], data, options);
    }

    clearNetwork(){
        $(this.refs.network).empty();
    }

    componentDidMount(){
        this.drawNetwork();
    }

    componentWillUnmount(){
        this.clearNetwork();
    }

    render(){
        return <div ref='network'></div>;
    }
}

Network.propTypes = {
    network: React.PropTypes.object.isRequired,
};

export default Network;
