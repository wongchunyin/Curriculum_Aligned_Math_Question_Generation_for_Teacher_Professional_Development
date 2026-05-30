import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  AppBar,
  Toolbar,
  Slider,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid
} from '@mui/material';
import { Assessment, Send, Add, Remove, Close, Percent, PictureAsPdf } from '@mui/icons-material';

function App() {
  const [scores, setScores] = useState({
    addition: '',
    subtraction: '',
    multiplication: '',
    division: '',
    passingScore: 50,
    year: 4,
    numQuestions: 5
  });

  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selectedQuestions, setSelectedQuestions] = useState([]);

  const getScoreLabel = (score, passingRequirement) => {
    if (score < passingRequirement) return "Weak";
    if (score < 75) return "Normal";
    return "Strong";
  };

  const isFormValid = () => {
    return scores.addition && scores.subtraction && scores.multiplication && scores.division;
  };

  const getSliderColor = (score) => {
    const numScore = parseInt(score) || 0;
    const passingScore = scores.passingScore;
    
    // Gradual color interpolation
    if (numScore <= passingScore) {
      // Red to Yellow (0 to passing score)
      const ratio = numScore / passingScore;
      const red = 244;
      const green = Math.round(67 + (152 * ratio)); // 67 to 219
      const blue = 54;
      return `rgb(${red}, ${green}, ${blue})`;
    } else {
      // Yellow to Green (passing score to 100)
      const ratio = Math.min((numScore - passingScore) / (100 - passingScore), 1);
      const red = Math.round(255 - (181 * ratio)); // 255 to 74
      const green = Math.round(193 + (82 * ratio)); // 193 to 175
      const blue = Math.round(7 + (73 * ratio)); // 7 to 80
      return `rgb(${red}, ${green}, ${blue})`;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const data = {
      transcript: {
        addition: parseInt(scores.addition),
        subtraction: parseInt(scores.subtraction),
        multiplication: parseInt(scores.multiplication),
        division: parseInt(scores.division)
      },
      scoreLabels: {
        addition: getScoreLabel(parseInt(scores.addition), scores.passingScore),
        subtraction: getScoreLabel(parseInt(scores.subtraction), scores.passingScore),
        multiplication: getScoreLabel(parseInt(scores.multiplication), scores.passingScore),
        division: getScoreLabel(parseInt(scores.division), scores.passingScore)
      },
      passingRequirement: scores.passingScore,
      year: scores.year,
      numQuestions: scores.numQuestions
    };

    console.log('=== SENDING TO BACKEND ===');
    console.log('Request URL:', 'http://127.0.0.1:5000/assess');
    console.log('Request Method:', 'POST');
    console.log('Request Headers:', { 'Content-Type': 'application/json' });
    console.log('Request Body:', JSON.stringify(data, null, 2));
    console.log('Raw Data Object:', data);
    console.log('========================');

    try {
      const res = await fetch('http://127.0.0.1:5000/assess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      
      const result = await res.json();
      setResponse(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderAssessmentResults = () => {
    if (!response) return null;

    const { questions } = response;
    const hasAnswers = questions.some(q => q.answer && q.answer.trim());
    
    const exportSelectedQuestions = () => {
      const selectedContent = selectedQuestions.map(index => {
        const questionData = questions[index];
        return `Question ${index + 1}: ${questionData.question}${questionData.answer ? `\n\nAnswer: ${questionData.answer}` : ''}`;
      }).join('\n\n---\n\n');
      
      const printWindow = window.open('', '_blank');
      printWindow.document.write(`
        <html>
          <head><title>Selected Math Questions</title></head>
          <body style="font-family: Arial, sans-serif; padding: 20px; white-space: pre-wrap;">
            <h1>Selected Math Questions</h1>
            ${selectedContent}
          </body>
        </html>
      `);
      printWindow.document.close();
      printWindow.print();
    };
    
    const handleQuestionSelect = (index, checked) => {
      if (checked) {
        setSelectedQuestions([...selectedQuestions, index]);
      } else {
        setSelectedQuestions(selectedQuestions.filter(i => i !== index));
      }
    };
    
    return (
      <Box sx={{ mt: 4 }}>
        <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Assessment color="primary" />
          Generated Materials
        </Typography>

        <TableContainer component={Paper} elevation={2}>
          <Table>
            <TableHead>
              <TableRow sx={{ backgroundColor: '#f5f5f5' }}>
                <TableCell sx={{ width: 60 }}>
                  <Checkbox
                    indeterminate={selectedQuestions.length > 0 && selectedQuestions.length < questions.length}
                    checked={questions.length > 0 && selectedQuestions.length === questions.length}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedQuestions(questions.map((_, index) => index));
                      } else {
                        setSelectedQuestions([]);
                      }
                    }}
                  />
                </TableCell>
                <TableCell sx={{ fontWeight: 600, color: 'primary.main' }}>
                  📝 Question
                </TableCell>
                {hasAnswers && (
                  <TableCell sx={{ fontWeight: 600, color: 'primary.main' }}>
                    ✅ Answer
                  </TableCell>
                )}
              </TableRow>
            </TableHead>
            <TableBody>
              {questions.map((questionData, index) => (
                <TableRow key={index} sx={{ 
                  '&:nth-of-type(odd)': { backgroundColor: '#fafafa' },
                  '&:hover': { backgroundColor: '#f0f0f0' }
                }}>
                  <TableCell sx={{ verticalAlign: 'top' }}>
                    <Checkbox
                      checked={selectedQuestions.includes(index)}
                      onChange={(e) => handleQuestionSelect(index, e.target.checked)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell sx={{ verticalAlign: 'top', width: hasAnswers ? '50%' : 'auto', p: 2 }}>
                    <Box sx={{ 
                      backgroundColor: '#fff3e0', 
                      p: 2, 
                      borderRadius: 2, 
                      border: '2px solid #ffb74d',
                      mb: 1
                    }}>
                      <Typography variant="subtitle1" sx={{ 
                        fontWeight: 700, 
                        color: '#e65100',
                        mb: 1,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                      }}>
                        📝 Question {index + 1}
                      </Typography>
                      <Typography 
                        variant="body1" 
                        sx={{ 
                          whiteSpace: 'pre-wrap', 
                          lineHeight: 1.7,
                          color: '#333',
                          fontSize: '1rem'
                        }}
                      >
                        {questionData.question || 'No question content'}
                      </Typography>
                    </Box>
                  </TableCell>
                  {hasAnswers && (
                    <TableCell sx={{ verticalAlign: 'top', width: '50%', p: 2 }}>
                      {questionData.answer && questionData.answer.trim() ? (
                        <Box sx={{ 
                          backgroundColor: '#e8f5e8', 
                          p: 2, 
                          borderRadius: 2, 
                          border: '2px solid #81c784'
                        }}>
                          <Typography variant="subtitle1" sx={{ 
                            fontWeight: 700, 
                            color: '#2e7d32',
                            mb: 1,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1
                          }}>
                            ✅ Answer
                          </Typography>
                          <Typography 
                            variant="body1" 
                            sx={{ 
                              whiteSpace: 'pre-wrap', 
                              lineHeight: 1.7,
                              color: '#333',
                              fontSize: '1rem'
                            }}
                          >
                            {questionData.answer}
                          </Typography>
                        </Box>
                      ) : (
                        <Box sx={{ 
                          backgroundColor: '#fafafa', 
                          p: 2, 
                          borderRadius: 2, 
                          border: '1px dashed #ccc',
                          textAlign: 'center'
                        }}>
                          <Typography variant="body2" color="text.secondary" sx={{ 
                            fontStyle: 'italic',
                            fontSize: '0.9rem'
                          }}>
                            ❌ No answer provided
                          </Typography>
                        </Box>
                      )}
                    </TableCell>
                  )}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        
        <Box sx={{ mt: 3, textAlign: 'center', display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="outlined"
            size="large"
            startIcon={<PictureAsPdf />}
            onClick={() => window.print()}
            sx={{ py: 1.5, px: 4, borderRadius: 2 }}
          >
            Export All as PDF
          </Button>
          {selectedQuestions.length > 0 && (
            <Button
              variant="contained"
              size="large"
              startIcon={<PictureAsPdf />}
              onClick={exportSelectedQuestions}
              sx={{ py: 1.5, px: 4, borderRadius: 2 }}
            >
              Export Selected ({selectedQuestions.length})
            </Button>
          )}
        </Box>
      </Box>
    );
  };

  return (
    <Box sx={{ flexGrow: 1, bgcolor: 'grey.50', minHeight: '100vh' }}>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Assessment sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Math Material Generation Platform
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
          <Typography variant="h4" component="h1" gutterBottom align="center" color="primary" sx={{ mb: 4 }}>
            Math Material Generator
          </Typography>

          <Box component="form" onSubmit={handleSubmit}>
            <Card elevation={1} sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Student Information & Assessment Criteria
                </Typography>
                <Box sx={{ mb: 3 }}>
                  <FormControl sx={{ minWidth: 200 }}>
                    <InputLabel>Year Level</InputLabel>
                    <Select
                      value={scores.year}
                      label="Year Level"
                      onChange={(e) => setScores({...scores, year: e.target.value})}
                    >
                      {[1,2,3,4,5,6,7,8,9,10].map((year) => (
                        <MenuItem key={year} value={year}>Year {year}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>
                    Minimum Passing Score
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, maxWidth: 400 }}>
                    <Tooltip title="Set the minimum score required to pass" arrow>
                      <Slider
                        value={scores.passingScore}
                        onChange={(e, value) => setScores({...scores, passingScore: value})}
                        min={0}
                        max={100}
                        step={5}
                        sx={{ flex: 1 }}
                        valueLabelDisplay="auto"
                        marks={[
                          { value: 0, label: '0' },
                          { value: 25, label: '25' },
                          { value: 50, label: '50' },
                          { value: 75, label: '75' },
                          { value: 100, label: '100' }
                        ]}
                      />
                    </Tooltip>
                    <Tooltip title="Enter exact passing score (0-100)" arrow>
                      <TextField
                        type="number"
                        value={scores.passingScore}
                        onChange={(e) => setScores({...scores, passingScore: parseInt(e.target.value) || 0})}
                        required
                        inputProps={{ min: 0, max: 100 }}
                        sx={{ width: 80 }}
                        size="small"
                        placeholder="0-100"
                        helperText="Score"
                      />
                    </Tooltip>
                  </Box>
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Students must score at or above this threshold to pass each math section
                  </Typography>
                </Box>
                <Typography variant="h6" color="primary" gutterBottom>
                  Math Skills Scores
                </Typography>
                <Grid container spacing={3}>
                  {[
                    { name: 'addition', icon: <Add sx={{ color: '#4caf50' }} /> },
                    { name: 'subtraction', icon: <Remove sx={{ color: '#f44336' }} /> },
                    { name: 'multiplication', icon: <Close sx={{ color: '#ff9800' }} /> },
                    { name: 'division', icon: <Percent sx={{ color: '#9c27b0' }} /> }
                  ].map(({ name, icon }) => (
                    <Grid item xs={12} sm={6} key={name}>
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle1" sx={{ textTransform: 'capitalize', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                          {icon}
                          {name} Score
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Tooltip title="Drag to adjust score or click to set value" arrow>
                            <Slider
                              value={parseInt(scores[name]) || 0}
                              onChange={(e, value) => setScores({...scores, [name]: value.toString()})}
                              min={0}
                              max={100}
                              step={1}
                              sx={{ 
                                flex: 1,
                                '& .MuiSlider-thumb': {
                                  backgroundColor: getSliderColor(scores[name]),
                                  transition: 'background-color 0.3s ease'
                                },
                                '& .MuiSlider-track': {
                                  backgroundColor: getSliderColor(scores[name]),
                                  transition: 'background-color 0.3s ease'
                                },
                                '& .MuiSlider-rail': {
                                  backgroundColor: '#e0e0e0'
                                }
                              }}
                              valueLabelDisplay="auto"
                              marks={[
                                { value: 0, label: '0' },
                                { value: scores.passingScore, label: scores.passingScore.toString() },
                                { value: 100, label: '100' }
                              ]}
                            />
                          </Tooltip>
                          <Tooltip title="Enter exact score (0-100)" arrow>
                            <TextField
                              type="number"
                              value={scores[name]}
                              onChange={(e) => setScores({...scores, [name]: e.target.value})}
                              required
                              inputProps={{ min: 0, max: 100 }}
                              sx={{ width: 80 }}
                              size="small"
                              placeholder="0-100"
                              helperText="Score"
                            />
                          </Tooltip>
                        </Box>
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>

            <Card elevation={1} sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" color="primary" gutterBottom>
                  Generation Settings
                </Typography>
                <Box>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>
                    Number of Questions
                  </Typography>
                  <TextField
                    type="number"
                    value={scores.numQuestions}
                    onChange={(e) => setScores({...scores, numQuestions: parseInt(e.target.value) || 1})}
                    required
                    inputProps={{ min: 1, max: 20 }}
                    sx={{ width: 120 }}
                    size="small"
                    placeholder="1-20"
                    helperText="Questions to generate"
                  />
                </Box>
              </CardContent>
            </Card>

            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={loading || !isFormValid()}
              startIcon={loading ? <CircularProgress size={20} /> : <Send />}
              sx={{ py: 1.5, px: 4, borderRadius: 2 }}
            >
              {loading ? 'Generating...' : 'Generate Materials'}
            </Button>
            {!isFormValid() && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Please fill in all math skill scores to generate materials
              </Typography>
            )}
          </Box>

          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              {error}
            </Alert>
          )}

          {renderAssessmentResults()}
        </Paper>
      </Container>
    </Box>
  );
}

export default App;