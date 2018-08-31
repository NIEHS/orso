import React from 'react';


class Network extends React.Component {

    drawNetwork(){
        var g = {
            nodes: this.props.network['nodes'],
            edges: this.props.network['edges'],
        };

        sigma.canvas.nodes.border = function(node, context, settings) {
            var prefix = settings('prefix') || '';

            context.fillStyle = node.color || settings('defaultNodeColor');
            context.beginPath();
            context.arc(
                node[prefix + 'x'],
                node[prefix + 'y'],
                node[prefix + 'size'],
                0,
                Math.PI * 2,
                true
            );
            context.closePath();
            context.fill();
            context.lineWidth = node.borderWidth || 1;
            context.strokeStyle = node.borderColor || 'rgb(0, 0, 0)';
            context.stroke();

            if (node.selected || 'False' == 'True') {
                context.beginPath();
                context.arc(
                    node[prefix + 'x'],
                    node[prefix + 'y'],
                    node[prefix + 'size'] * 1.2,
                    0,
                    Math.PI * 2,
                    true
                );
                context.closePath();
                context.lineWidth = node.borderWidth || 1;
                context.strokeStyle = node.borderColor || 'rgb(0, 0, 0)';
                context.stroke();
            }
        };

        var s = new sigma({
            graph: g,
            container: 'network',
            settings: {
                drawLabels: false,
                minNodeSize: 2,
                maxNodeSize: 24,
                defaultNodeType: 'border',
            },
            renderer: {
                container: document.getElementById('network'),
                type: 'canvas',
            },
        });

        var cam = s.camera;

        var n = s.graph.nodes('center');
        if (typeof(n) != "undefined") {
            cam.goTo({
                x: n[cam.readPrefix + 'x'],
                y: n[cam.readPrefix + 'y'],
                ratio: this.props.network['camera']['zoom_ratio'],
            })
        }
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
        return <div
            style={{height:'100%', width:"100%", border: '1px solid black'}}
            ref='network' id='network'>
        </div>;
    }
}

Network.propTypes = {
    network: React.PropTypes.object.isRequired,
};

export default Network;
