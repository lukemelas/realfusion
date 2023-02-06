import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements, video_url } from 'data'


const Index = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Video */}
    <Container w="90vw" h="50.6vw" maxW="50rem" maxH="25rem" mb="3rem">
      <iframe
        width="100%" height="100%"
        src={video_url}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container>

    {/* Main */}
    <Container w="100%" maxW="60rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontWeight="light" fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">{abstract}</Text>

      {/* Example */}
      <Heading fontWeight="light" fontSize="2xl" pb="1rem">Examples</Heading>
      <video autoPlay loop src={`${process.env.BASE_PATH || ""}/videos/examples.mp4`} />
      <Text maxW="50rem" margin="auto" pt="0.5rem" pb="0.5rem" fontSize="small">
        <Text as="span" fontWeight="bold">Examples.</Text> RealFusion reconstructions from a single input view.
      </Text>

      {/* Another Section */}
      <Heading fontWeight="light" fontSize="2xl" pt="2rem" pb="1rem" id="dataset">Diagram</Heading>
      <Text ><img src={`${process.env.BASE_PATH || ""}/images/method-diagram-v3.png`} /></Text>
      <Text maxW="50rem" margin="auto" pt="0.5rem" pb="0.5rem" fontSize="small">
        <Text as="span" fontWeight="bold">Method diagram.</Text> Our method optimizes a neural radiance field using two objectives simultaneously: a reconstruction objective and a prior objective. The reconstruction objective ensures that the radiance field resembles the input image from a specific, fixed view. The prior objective uses a large pre-trained diffusion model to ensure that the radiance field looks like the given object from randomly sampled novel viewpoints. The key to making this process work well is to condition the diffusion model on a prompt with a custom token <Text as="span" fontWeight="bold">&lt;e&gt;</Text>, which is generated prior to reconstruction using single-image textual inversion.
      </Text>

      {/* Citation */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px" overflow="scroll" whiteSpace="nowrap">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
        &#125;
      </Code>

      {/* Acknowledgements */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        {acknowledgements}
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
    
  </Container >
)

export default Index
