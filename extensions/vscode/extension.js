const vscode = require('vscode');
const axios = require('axios');

function activate(context) {
    console.log('DevSage extension is now active!');

    // Register command for asking DevSage
    let askCommand = vscode.commands.registerCommand('devsage.ask', async function () {
        const prompt = await vscode.window.showInputBox({
            placeHolder: 'Ask DevSage anything about your code...',
            prompt: 'DevSage Assistant'
        });
        
        if (prompt) {
            try {
                const response = await queryDevSage(prompt, {
                    currentFile: vscode.window.activeTextEditor?.document.fileName,
                    workspace: vscode.workspace.rootPath
                });
                
                vscode.window.showInformationMessage(response.result || response.error);
            } catch (error) {
                vscode.window.showErrorMessage(`DevSage error: ${error.message}`);
            }
        }
    });

    context.subscriptions.push(askCommand);
}

async function queryDevSage(prompt, context) {
    const config = vscode.workspace.getConfiguration('devsage');
    const apiUrl = config.get('apiUrl') || 'http://localhost:8000';
    
    const response = await axios.post(`${apiUrl}/v1/agent/query`, {
        prompt,
        context
    }, {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer devsage_token_123`
        }
    });
    
    return response.data;
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
}